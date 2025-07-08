<?php
// customer_side/payment/payment_pin.php - No Account Locking Version
require_once '../config.php';
require_once '../encrypt.php';

// Start session and check for payment session
SessionManager::start();

// Check if we have a valid payment session from gateway
if (!isset($_SESSION['gateway_payment'])) {
    header("Location: ../../gateway/checkout.php");
    exit();
}

// ✅ ENHANCED FLOW PROTECTION
// Check if face verification was completed AND is still valid
if (!isset($_SESSION['payment_user']) || !$_SESSION['payment_user']['face_verified']) {
    header("Location: payment_scan.php");
    exit();
}

// ✅ Check if face verification is too old (30 minutes timeout)
$face_verified_at = $_SESSION['payment_user']['face_verified_at'] ?? 0;
if ((time() - $face_verified_at) > 1800) { // 30 minutes
    // Face verification expired - clear session and restart
    unset($_SESSION['payment_user']);
    SecurityUtils::logSecurityEvent($_SESSION['payment_user']['user_id'] ?? 'unknown', 'payment_face_expired', 'failure', [
        'expired_duration' => time() - $face_verified_at
    ]);
    header("Location: payment_scan.php");
    exit();
}

// ✅ Check if user is trying to access PIN step without completing face verification
if ($_SESSION['payment_user']['payment_step'] !== 'pin_entry') {
    header("Location: payment_scan.php");
    exit();
}

// ✅ Prevent PIN step from being accessed after completion
if (isset($_SESSION['payment_user']['pin_verified']) && $_SESSION['payment_user']['pin_verified']) {
    // PIN already verified - redirect to payment processing
    header("Location: payment_process.php");
    exit();
}

$payment_data = $_SESSION['gateway_payment'];

// Get merchant main URL for cancel redirects
$merchant_main_url = '';
if (isset($_SESSION['gateway_payment']['merchant_id'])) {
    try {
        $conn = dbConnect();
        $stmt = $conn->prepare("SELECT return_url FROM merchants WHERE merchant_id = ?");
        $stmt->bind_param("s", $_SESSION['gateway_payment']['merchant_id']);
        $stmt->execute();
        $result = $stmt->get_result();
        $merchant = $result->fetch_assoc();
        $merchant_main_url = $merchant['return_url'] ?? '';
        $stmt->close();
        $conn->close();
    } catch (Exception $e) {
        error_log("Error getting merchant URL: " . $e->getMessage());
    }
}
// Add to payment_data for JavaScript access
$payment_data['merchant_main_url'] = $merchant_main_url;

$payment_user = $_SESSION['payment_user'];
$error = "";
$success = "";

// ✅ INITIALIZE PIN ATTEMPTS
if (!isset($_SESSION['payment_pin_attempts'])) {
    $_SESSION['payment_pin_attempts'] = 0;
}

$attempts_used = $_SESSION['payment_pin_attempts'];
$attempts_remaining = 5 - $attempts_used;

// ✅ NEW: Function to handle payment failure redirects
function redirectToPaymentFailed($payment_data, $error_message) {
    // Clear payment session
    unset($_SESSION['payment_user']);
    unset($_SESSION['payment_pin_attempts']);
    
    // Store the redirect data in session for the failed page
    $_SESSION['payment_redirect_data'] = [
        'status' => 'failed',
        'order_id' => $payment_data['order_id'],
        'amount' => $payment_data['amount'],
        'currency' => $payment_data['currency'],
        'merchant_name' => $payment_data['merchant_name'],
        'merchant_url' => $payment_data['merchant_main_url'] ?? '',
        'error' => $error_message
    ];
    
    // Simple PHP redirect - much more reliable
    header("Location: payment_failed.php");
    exit();
}

// ✅ Check if max attempts already reached (before any POST processing)
if ($attempts_used >= 5) {
    SecurityUtils::logSecurityEvent($payment_user['user_id'], 'payment_pin_max_attempts_reached', 'failure', [
        'payment_amount' => $payment_data['amount'],
        'merchant_id' => $payment_data['merchant_id'],
        'order_id' => $payment_data['order_id'],
        'attempts_used' => $attempts_used
    ]);
    
    redirectToPaymentFailed($payment_data, 'Payment cancelled due to too many incorrect PIN attempts');
}

// Handle PIN verification
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $action = $_POST['action'] ?? '';
    
    if ($action === 'verify_pin') {
        // CSRF validation
        if (!validateCSRFToken($_POST['csrf_token'] ?? '')) {
            $error = "Invalid request. Please try again.";
        } else {
            // ✅ Check if face verification is still valid
            if ((time() - $face_verified_at) > 1800) { // 30 minutes
                $error = "Face verification expired. Please restart the payment process.";
                unset($_SESSION['payment_user']);
                unset($_SESSION['payment_pin_attempts']);
                echo "<script>
                    setTimeout(() => {
                        window.location.href = 'payment_scan.php';
                    }, 3000);
                </script>";
                // Don't exit here, let the page render with the error
            } 
            else {
                $entered_pin = $_POST['pin'] ?? '';
                
                if (empty($entered_pin)) {
                    $error = "Please enter your PIN.";
                } elseif (strlen($entered_pin) !== 6 || !ctype_digit($entered_pin)) {
                    $error = "PIN must be exactly 6 digits.";
                    SecurityUtils::logSecurityEvent($payment_user['user_id'], 'payment_invalid_pin_format', 'failure', [
                        'payment_amount' => $payment_data['amount'],
                        'merchant_id' => $payment_data['merchant_id'],
                        'order_id' => $payment_data['order_id']
                    ]);
                } else {
                    try {
                        $conn = dbConnect();
                        
                        // ✅ Get PIN and account status
                        $stmt = $conn->prepare("
                            SELECT pin_code, account_status
                            FROM users 
                            WHERE id = ? AND account_status = 'active'
                        ");
                        $stmt->bind_param("i", $payment_user['id']);
                        $stmt->execute();
                        $result = $stmt->get_result();
                        $user_data = $result->fetch_assoc();
                        $stmt->close();
                        
                        if (!$user_data) {
                            $error = "Account not found or not active.";
                            SecurityUtils::logSecurityEvent($payment_user['user_id'], 'payment_pin_account_error', 'failure');
                            $conn->close();
                            redirectToPaymentFailed($payment_data, 'Account not found or not active');
                        } else {
                            // ✅ Verify PIN
                            if (password_verify($entered_pin, $user_data['pin_code'])) {
                                // ✅ SUCCESS - Clear PIN attempts and proceed
                                unset($_SESSION['payment_pin_attempts']);
                                
                                // ✅ UPDATE SESSION WITH FLOW PROTECTION
                                $_SESSION['payment_user']['pin_verified'] = true;
                                $_SESSION['payment_user']['pin_verified_at'] = time();
                                $_SESSION['payment_user']['payment_step'] = 'processing';
                                
                                SecurityUtils::logSecurityEvent($payment_user['user_id'], 'payment_pin_success', 'success', [
                                    'payment_amount' => $payment_data['amount'],
                                    'merchant_id' => $payment_data['merchant_id'],
                                    'order_id' => $payment_data['order_id'],
                                    'attempts_used' => $attempts_used + 1
                                ]);
                                
                                $success = "PIN verified! Processing payment...";
                            } else {
                                // ❌ PIN FAILED - Increment attempts
                                $_SESSION['payment_pin_attempts'] = $attempts_used + 1;
                                $new_attempts_used = $_SESSION['payment_pin_attempts'];
                                $new_attempts_remaining = 5 - $new_attempts_used;
                                
                                // Update display variables
                                $attempts_used = $new_attempts_used;
                                $attempts_remaining = $new_attempts_remaining;
                                
                                SecurityUtils::logSecurityEvent($payment_user['user_id'], 'payment_pin_failed', 'failure', [
                                    'payment_amount' => $payment_data['amount'],
                                    'merchant_id' => $payment_data['merchant_id'],
                                    'order_id' => $payment_data['order_id'],
                                    'attempts_used' => $new_attempts_used,
                                    'attempts_remaining' => $new_attempts_remaining
                                ]);
                                
                                // ✅ Check if this was the final attempt
                                if ($new_attempts_remaining <= 0) {
                                    $conn->close();
                                    redirectToPaymentFailed($payment_data, 'Maximum PIN attempts exceeded for this payment');
                                } else {
                                    $error = "Incorrect PIN. $new_attempts_remaining attempts remaining for this payment.";
                                }
                            }
                        }
                        
                        $conn->close();
                        
                    } catch (Exception $e) {
                        error_log("Payment PIN verification error: " . $e->getMessage());
                        redirectToPaymentFailed($payment_data, 'Payment authorization failed due to system error');
                    }
                }
            }
        }
    }
}

// ✅ Final check: if attempts remaining is 0 or less, redirect
if ($attempts_remaining <= 0) {
    redirectToPaymentFailed($payment_data, 'Payment cancelled due to too many incorrect PIN attempts');
}

$csrf_token = generateCSRFToken();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PIN Authorization - FacePay</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary: #667eea;
            --success: #38a169;
            --warning: #ed8936;
            --error: #e53e3e;
            --text-dark: #2d3748;
            --text-medium: #4a5568;
            --text-light: #718096;
            --border: #e2e8f0;
            --bg-light: #f7fafc;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: var(--text-dark);
            line-height: 1.6;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            width: 100%;
            max-width: 600px;
        }
        
        .pin-container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            text-align: center;
            animation: fadeUp 0.6s ease-out;
        }
        
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .pin-header {
            margin-bottom: 35px;
        }
        
        .pin-logo {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, var(--success), #48bb78);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
        }
        
        .pin-logo i {
            font-size: 32px;
            color: white;
        }
        
        h2 {
            color: var(--text-dark);
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        
        .subtitle {
            color: var(--text-light);
            font-size: 16px;
            margin-bottom: 30px;
        }
        
        .user-info {
            background: var(--bg-light);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 25px;
            border: 2px solid #c6f6d5;
        }
        
        .verification-status {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .status-icon {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--success);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
        }
        
        .payment-summary {
            background: #fff9f0;
            border: 2px solid #fbd38d;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
        }
        
        .payment-amount {
            font-size: 32px;
            font-weight: 700;
            color: var(--warning);
            margin-bottom: 5px;
        }
        
        .payment-merchant {
            font-size: 16px;
            color: var(--text-medium);
            font-weight: 600;
        }
        
        .attempts-indicator {
            background: #fff5f5;
            border: 2px solid #fed7d7;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        .attempts-indicator.warning {
            background: #fffbeb;
            border-color: #fbd38d;
        }
        
        .attempts-indicator.normal {
            background: #f0fff4;
            border-color: #c6f6d5;
        }
        
        .attempts-text {
            font-weight: 600;
            color: var(--text-dark);
        }
        
        .pin-section {
            margin: 30px 0;
        }
        
        .pin-instruction {
            font-size: 18px;
            color: var(--text-dark);
            font-weight: 600;
            margin-bottom: 25px;
        }
        
        .pin-inputs {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .pin-digit {
            width: 50px;
            height: 60px;
            border: 3px solid var(--border);
            border-radius: 12px;
            text-align: center;
            font-size: 24px;
            font-weight: 700;
            background: white;
            transition: all 0.3s ease;
            outline: none;
        }
        
        .pin-digit.filled {
            background: var(--success);
            color: white;
            border-color: var(--success);
        }
        
        .keypad {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            max-width: 300px;
            margin: 25px auto;
        }
        
        .keypad-btn {
            width: 80px;
            height: 80px;
            border: 2px solid var(--border);
            border-radius: 50%;
            background: white;
            font-size: 24px;
            font-weight: 700;
            color: var(--text-dark);
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .keypad-btn:hover {
            background: var(--success);
            color: white;
            border-color: var(--success);
            transform: scale(1.05);
        }
        
        .keypad-btn:active {
            transform: scale(0.95);
        }
        
        .keypad-btn.special {
            background: var(--bg-light);
            font-size: 16px;
        }
        
        .keypad-btn.special:hover {
            background: var(--error);
        }
        
        .message {
            padding: 15px 18px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 500;
        }
        
        .success-message {
            background-color: #f0fff4;
            color: var(--success);
            border: 1px solid #c6f6d5;
        }
        
        .error-message {
            background-color: #fff5f5;
            color: var(--error);
            border: 1px solid #fed7d7;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 30px;
        }
        
        .btn {
            padding: 15px 25px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            text-decoration: none;
            min-width: 160px;
            justify-content: center;
        }
        
        .btn-success {
            background: var(--success);
            color: white;
        }
        
        .btn-success:hover {
            background: #2f855a;
            transform: translateY(-2px);
        }
        
        .btn-success:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-warning {
            background: #95a5a6;
            color: white;
            border: 2px solid #95a5a6;
        }
        
        .btn-warning:hover {
            background: #7f8c8d;
            border-color: #7f8c8d;
        }
        
        .security-notice {
            background: #f0f8ff;
            border: 1px solid #bfdbfe;
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            font-size: 13px;
            color: var(--text-medium);
            text-align: left;
        }
        
        .security-notice i {
            color: var(--primary);
            margin-right: 8px;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .pin-container {
                padding: 25px 20px;
            }
            
            .pin-digit {
                width: 45px;
                height: 55px;
                font-size: 20px;
            }
            
            .keypad {
                max-width: 250px;
                gap: 10px;
            }
            
            .keypad-btn {
                width: 70px;
                height: 70px;
                font-size: 20px;
            }
            
            .controls {
                flex-direction: column;
                align-items: center;
            }
            
            .btn {
                width: 100%;
                max-width: 250px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="pin-container">
            <div class="pin-header">
                <div class="pin-logo">
                    <i class="fas fa-shield-alt"></i>
                </div>
                <h2>Enter Your PIN</h2>
                <p class="subtitle">Complete your payment authorization with your 6-digit PIN</p>
            </div>
            
            <div class="user-info">
                <div class="verification-status">
                    <div class="status-icon">
                        <i class="fas fa-check"></i>
                    </div>
                    <span style="color: var(--success); font-weight: 600;">Face Verified: <?= htmlspecialchars($payment_user['full_name']) ?></span>
                </div>
            </div>
            
            <div class="payment-summary">
                <div class="payment-amount">
                    <?= $payment_data['currency'] ?> <?= number_format($payment_data['amount'], 2) ?>
                </div>
                <div class="payment-merchant">
                    to <?= htmlspecialchars($payment_data['merchant_name']) ?>
                </div>
            </div>
            
            <?php if ($error): ?>
                <div class="message error-message">
                    <i class="fas fa-exclamation-circle"></i>
                    <?= htmlspecialchars($error) ?>
                </div>
            <?php else: ?>
                <!-- ✅ Show attempts remaining only when no error -->
                <div class="attempts-indicator <?= $attempts_remaining <= 2 ? 'warning' : ($attempts_remaining <= 1 ? '' : 'normal') ?>">
                    <div class="attempts-text">
                        <i class="fas fa-exclamation-triangle"></i>
                        <?= $attempts_remaining ?> attempts remaining for this payment
                    </div>
                </div>
            <?php endif; ?>
            
            <?php if ($success): ?>
                <div class="message success-message">
                    <i class="fas fa-check-circle"></i>
                    <?= htmlspecialchars($success) ?>
                </div>
            <?php endif; ?>
            
            <div class="pin-section">
                <div class="pin-instruction">
                    Enter your 6-digit security PIN
                </div>
                
                <form method="POST" id="pinForm">
                    <input type="hidden" name="csrf_token" value="<?= $csrf_token ?>">
                    <input type="hidden" name="action" value="verify_pin">
                    <input type="hidden" name="pin" id="hiddenPin">
                    
                    <div class="pin-inputs" id="pinInputs">
                        <input type="password" class="pin-digit" maxlength="1" data-index="0" readonly>
                        <input type="password" class="pin-digit" maxlength="1" data-index="1" readonly>
                        <input type="password" class="pin-digit" maxlength="1" data-index="2" readonly>
                        <input type="password" class="pin-digit" maxlength="1" data-index="3" readonly>
                        <input type="password" class="pin-digit" maxlength="1" data-index="4" readonly>
                        <input type="password" class="pin-digit" maxlength="1" data-index="5" readonly>
                    </div>
                </form>
            </div>
            
            <div class="keypad" id="keypad">
                <button class="keypad-btn" data-digit="1">1</button>
                <button class="keypad-btn" data-digit="2">2</button>
                <button class="keypad-btn" data-digit="3">3</button>
                <button class="keypad-btn" data-digit="4">4</button>
                <button class="keypad-btn" data-digit="5">5</button>
                <button class="keypad-btn" data-digit="6">6</button>
                <button class="keypad-btn" data-digit="7">7</button>
                <button class="keypad-btn" data-digit="8">8</button>
                <button class="keypad-btn" data-digit="9">9</button>
                <button class="keypad-btn special" data-action="clear">
                    <i class="fas fa-undo"></i>
                </button>
                <button class="keypad-btn" data-digit="0">0</button>
                <button class="keypad-btn special" data-action="backspace">
                    <i class="fas fa-backspace"></i>
                </button>
            </div>
            
            <div class="controls">
                <button class="btn btn-success" id="submitBtn" disabled>
                    <i class="fas fa-lock"></i>
                    Authorize Payment
                </button>
                
                <a href="javascript:void(0)" class="btn btn-warning" onclick="cancelPayment()">
                    <i class="fas fa-times"></i>
                    Cancel Payment
                </a>
            </div>
            
            <div class="security-notice">
                <i class="fas fa-info-circle"></i>
                <strong>Payment Security:</strong> You have <?= $attempts_remaining ?> attempts for this payment. After 5 failed attempts, the payment will be cancelled for security.
            </div>
        </div>
    </div>

    <script>
        let currentPin = '';
        const maxPinLength = 6;
        
        // ✅ FLOW PROTECTION - Prevent browser back button abuse
        window.history.pushState(null, "", window.location.href);
        window.onpopstate = function() {
            window.history.pushState(null, "", window.location.href);
        };
        
        // Initialize PIN input
        document.addEventListener('DOMContentLoaded', function() {
            setupKeypad();
            
            const submitBtn = document.getElementById('submitBtn');
            if (submitBtn) {
                submitBtn.addEventListener('click', submitPin);
            }
        });
        
        // Setup virtual keypad
        function setupKeypad() {
            const keypadBtns = document.querySelectorAll('.keypad-btn');
            
            keypadBtns.forEach((btn, index) => {
                btn.addEventListener('click', function(e) {
                    e.preventDefault();
                    const digit = this.getAttribute('data-digit');
                    const action = this.getAttribute('data-action');
                    
                    if (digit !== null) {
                        addDigit(digit);
                    } else if (action === 'backspace') {
                        removeLastDigit();
                    } else if (action === 'clear') {
                        clearPin();
                    }
                });
            });
        }
        
        // Add digit to PIN
        function addDigit(digit) {
            if (currentPin.length < maxPinLength) {
                currentPin += digit;
                updatePinDisplay();
                updateSubmitButton();
                
                // Add visual feedback
                const currentIndex = currentPin.length - 1;
                const pinInput = document.querySelector(`.pin-digit[data-index="${currentIndex}"]`);
                if (pinInput) {
                    pinInput.value = '•';
                    pinInput.classList.add('filled');
                }
            }
        }
        
        // Remove last digit
        function removeLastDigit() {
            if (currentPin.length > 0) {
                const lastIndex = currentPin.length - 1;
                const pinInput = document.querySelector(`.pin-digit[data-index="${lastIndex}"]`);
                if (pinInput) {
                    pinInput.value = '';
                    pinInput.classList.remove('filled');
                }
                
                currentPin = currentPin.slice(0, -1);
                updateSubmitButton();
            }
        }
        
        // Clear entire PIN
        function clearPin() {
            currentPin = '';
            const pinInputs = document.querySelectorAll('.pin-digit');
            pinInputs.forEach(input => {
                input.value = '';
                input.classList.remove('filled');
            });
            updateSubmitButton();
        }
        
        // Update PIN display
        function updatePinDisplay() {
            const pinInputs = document.querySelectorAll('.pin-digit');
            pinInputs.forEach((input, index) => {
                if (index < currentPin.length) {
                    input.value = '•';
                    input.classList.add('filled');
                } else {
                    input.value = '';
                    input.classList.remove('filled');
                }
            });
        }
        
        // Update submit button state
        function updateSubmitButton() {
            const submitBtn = document.getElementById('submitBtn');
            if (submitBtn) {
                const isEnabled = currentPin.length === maxPinLength;
                submitBtn.disabled = !isEnabled;
            }
        }
        
        // Submit PIN
        function submitPin() {
            if (currentPin.length === maxPinLength) {
                // Set PIN in hidden field
                document.getElementById('hiddenPin').value = currentPin;
                
                // Clear PIN from memory for security
                currentPin = '';
                clearPin();
                
                // Submit form
                document.getElementById('pinForm').submit();
            }
        }
        
        // Allow keyboard input
        document.addEventListener('keydown', function(e) {
            if (e.key >= '0' && e.key <= '9') {
                e.preventDefault();
                addDigit(e.key);
            } else if (e.key === 'Backspace') {
                e.preventDefault();
                removeLastDigit();
            } else if (e.key === 'Enter' && currentPin.length === maxPinLength) {
                e.preventDefault();
                submitPin();
            } else if (e.key === 'Escape') {
                e.preventDefault();
                clearPin();
            }
        });
        
        // Cancel payment function
        function cancelPayment() {
            if (confirm('Are you sure you want to cancel this payment?')) {
                // Clear PIN from memory for security
                currentPin = '';
                clearPin();
                
                // Clear only client-side UI data
                if (typeof(Storage) !== "undefined") {
                    sessionStorage.removeItem('pin_entered');
                    sessionStorage.removeItem('pin_attempts');
                }
                
                // Use POST form to redirect to payment_failed.php
                const form = document.createElement('form');
                form.method = 'POST';
                form.action = 'payment_failed.php';
                
                const fields = {
                    'status': 'cancelled',
                    'order_id': '<?= htmlspecialchars($payment_data['order_id']) ?>',
                    'amount': '<?= $payment_data['amount'] ?>',
                    'currency': '<?= htmlspecialchars($payment_data['currency']) ?>',
                    'merchant_name': '<?= htmlspecialchars($payment_data['merchant_name']) ?>',
                    'merchant_url': '<?= htmlspecialchars($payment_data['merchant_main_url'] ?? '') ?>',
                    'error': 'Payment cancelled by user during PIN entry'
                };
                
                Object.keys(fields).forEach(key => {
                    const input = document.createElement('input');
                    input.type = 'hidden';
                    input.name = key;
                    input.value = fields[key];
                    form.appendChild(input);
                });
                
                document.body.appendChild(form);
                form.submit();
            }
        }
        
        // Handle successful PHP response
        <?php if ($success): ?>
        setTimeout(() => {
            window.location.href = 'payment_process.php';
        }, 3000);
        <?php endif; ?>
        
        // Security: Clear PIN from memory on page unload
        window.addEventListener('beforeunload', function() {
            currentPin = '';
            clearPin();
        });
    </script>
</body>
</html>