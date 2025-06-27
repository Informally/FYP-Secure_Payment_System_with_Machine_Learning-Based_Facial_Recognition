<?php
// customer_side/payment/payment_pin.php - PIN Entry for Payment Authorization
require_once '../config.php';
require_once '../encrypt.php';

// Start session and check for payment session
SessionManager::start();

// Check if we have a valid payment session from gateway
if (!isset($_SESSION['gateway_payment'])) {
    header("Location: ../../gateway/checkout.php");
    exit();
}

// Check if face verification was completed
if (!isset($_SESSION['payment_user']) || !$_SESSION['payment_user']['face_verified']) {
    header("Location: payment_scan.php");
    exit();
}

$payment_data = $_SESSION['gateway_payment'];
$payment_user = $_SESSION['payment_user'];
$error = "";
$success = "";

// Handle PIN verification
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $action = $_POST['action'] ?? '';
    
    if ($action === 'verify_pin') {
        // CSRF validation
        if (!validateCSRFToken($_POST['csrf_token'] ?? '')) {
            $error = "Invalid request. Please try again.";
        } else {
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
                    
                    // Get user's PIN and check account status - Using your existing table structure
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
                    } else {
                        // Simple PIN verification - direct comparison with pin_code field
                        if (password_verify($entered_pin, $user_data['pin_code'])) {
                            // PIN correct - proceed with payment
                            $_SESSION['payment_user']['pin_verified'] = true;
                            $_SESSION['payment_user']['pin_verified_at'] = time();
                            
                            SecurityUtils::logSecurityEvent($payment_user['user_id'], 'payment_pin_success', 'success', [
                                'payment_amount' => $payment_data['amount'],
                                'merchant_id' => $payment_data['merchant_id'],
                                'order_id' => $payment_data['order_id'],
                                'face_similarity' => $payment_user['face_similarity'],
                                'liveness_score' => $payment_user['liveness_score']
                            ]);
                            
                            $success = "PIN verified! Processing payment...";
                            
                            // JavaScript will handle redirect to payment processing
                        } else {
                            // PIN incorrect
                            $error = "Incorrect PIN. Please try again.";
                            
                            SecurityUtils::logSecurityEvent($payment_user['user_id'], 'payment_pin_failed', 'failure', [
                                'payment_amount' => $payment_data['amount'],
                                'merchant_id' => $payment_data['merchant_id']
                            ]);
                        }
                    }
                    
                    $conn->close();
                    
                } catch (Exception $e) {
                    error_log("Payment PIN verification error: " . $e->getMessage());
                    $error = "Payment authorization failed due to system error.";
                }
            }
        }
    }
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
            --primary-light: #e3f2fd;
            --primary-dark: #5a67d8;
            --success: #38a169;
            --warning: #ed8936;
            --error: #e53e3e;
            --text-dark: #2d3748;
            --text-medium: #4a5568;
            --text-light: #718096;
            --border: #e2e8f0;
            --bg-light: #f7fafc;
            --shadow: rgba(0,0,0,0.1);
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
            position: relative;
            overflow: hidden;
        }
        
        .pin-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--success), #667eea);
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
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
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
        
        /* User Info Section */
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
            margin-bottom: 15px;
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
        
        .user-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            text-align: left;
        }
        
        .detail-item {
            display: flex;
            flex-direction: column;
        }
        
        .detail-label {
            font-size: 12px;
            color: var(--text-light);
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .detail-value {
            font-size: 14px;
            color: var(--text-dark);
            font-weight: 600;
        }
        
        /* Payment Summary */
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
        
        /* PIN Input Section */
        .pin-section {
            margin: 30px 0;
        }
        
        .pin-instruction {
            font-size: 18px;
            color: var(--text-dark);
            font-weight: 600;
            margin-bottom: 25px;
        }
        
        .pin-input-container {
            margin: 25px 0;
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
        
        .pin-digit:focus {
            border-color: var(--success);
            box-shadow: 0 0 0 3px rgba(56, 161, 105, 0.1);
            transform: scale(1.05);
        }
        
        .pin-digit.filled {
            background: var(--success);
            color: white;
            border-color: var(--success);
        }
        
        /* Virtual Keypad */
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
        
        /* Messages */
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
        
        /* Controls */
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
        
        .btn-secondary {
            background: white;
            color: var(--text-medium);
            border: 2px solid var(--border);
        }
        
        .btn-secondary:hover {
            background: var(--bg-light);
            border-color: var(--primary);
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
        
        /* Security Notice */
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
        
        /* Loading State */
        .loading {
            display: none;
            flex-direction: column;
            align-items: center;
            gap: 15px;
            margin: 20px 0;
        }
        
        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #e2e8f0;
            border-top: 4px solid var(--success);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .pin-container {
                padding: 25px 20px;
            }
            
            .user-details {
                grid-template-columns: 1fr;
                gap: 10px;
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
            
            <!-- User Verification Status -->
            <div class="user-info">
                <div class="verification-status">
                    <div class="status-icon">
                        <i class="fas fa-check"></i>
                    </div>
                    <span style="color: var(--success); font-weight: 600;">Face Verified: <?= htmlspecialchars($payment_user['full_name']) ?></span>
                </div>
                <!-- <div class="user-details">
                    <div class="detail-item">
                        <div class="detail-label">Username</div>
                        <div class="detail-value"><?= htmlspecialchars($payment_user['username']) ?></div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Similarity Score</div>
                        <div class="detail-value"><?= number_format($payment_user['face_similarity'] * 100, 1) ?>%</div>
                    </div>
                </div> -->
            </div>
            
            <!-- Payment Summary -->
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
            <?php endif; ?>
            
            <?php if ($success): ?>
                <div class="message success-message">
                    <i class="fas fa-check-circle"></i>
                    <?= htmlspecialchars($success) ?>
                </div>
            <?php endif; ?>
            
            <!-- PIN Input Section -->
            <div class="pin-section">
                <div class="pin-instruction">
                    Enter your 6-digit security PIN
                </div>
                
                <form method="POST" id="pinForm">
                    <input type="hidden" name="csrf_token" value="<?= $csrf_token ?>">
                    <input type="hidden" name="action" value="verify_pin">
                    <input type="hidden" name="pin" id="hiddenPin">
                    
                    <div class="pin-input-container">
                        <div class="pin-inputs" id="pinInputs">
                            <input type="password" class="pin-digit" maxlength="1" data-index="0" readonly>
                            <input type="password" class="pin-digit" maxlength="1" data-index="1" readonly>
                            <input type="password" class="pin-digit" maxlength="1" data-index="2" readonly>
                            <input type="password" class="pin-digit" maxlength="1" data-index="3" readonly>
                            <input type="password" class="pin-digit" maxlength="1" data-index="4" readonly>
                            <input type="password" class="pin-digit" maxlength="1" data-index="5" readonly>
                        </div>
                    </div>
                </form>
            </div>
            
            <!-- Virtual Keypad -->
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
            
            <!-- Loading State -->
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div>Processing payment...</div>
            </div>
            
            <!-- Security Notice -->
            <div class="security-notice">
                <i class="fas fa-info-circle"></i>
                <strong>Security Notice:</strong> Your PIN is verified securely for payment authorization. 
                Please ensure you enter the correct PIN associated with your account.
            </div>
            
            <!-- Controls -->
            <div class="controls">
                <button class="btn btn-success" id="submitBtn" disabled>
                    <i class="fas fa-lock"></i>
                    Authorize Payment
                </button>
                
                <a href="payment_scan.php" class="btn btn-secondary">
                    <i class="fas fa-arrow-left"></i>
                    Back to Face Scan
                </a>
                
                <a href="javascript:void(0)" class="btn btn-warning" onclick="cancelPayment()">
                    <i class="fas fa-times"></i>
                    Cancel Payment
                </a>
            </div>
        </div>
    </div>

    <script>
        let currentPin = '';
        const maxPinLength = 6;
        
        // Initialize PIN input
        document.addEventListener('DOMContentLoaded', function() {
            setupKeypad();
            document.getElementById('submitBtn').addEventListener('click', submitPin);
        });
        
        // Setup virtual keypad
        function setupKeypad() {
            const keypadBtns = document.querySelectorAll('.keypad-btn');
            
            keypadBtns.forEach(btn => {
                btn.addEventListener('click', function() {
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
                pinInput.value = '•';
                pinInput.classList.add('filled');
            }
        }
        
        // Remove last digit
        function removeLastDigit() {
            if (currentPin.length > 0) {
                const lastIndex = currentPin.length - 1;
                const pinInput = document.querySelector(`.pin-digit[data-index="${lastIndex}"]`);
                pinInput.value = '';
                pinInput.classList.remove('filled');
                
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
            submitBtn.disabled = currentPin.length !== maxPinLength;
        }
        
        // Submit PIN
        function submitPin() {
            if (currentPin.length === maxPinLength) {
                // Show loading state
                document.getElementById('loading').style.display = 'flex';
                document.getElementById('submitBtn').disabled = true;
                
                // Set PIN in hidden field and submit
                document.getElementById('hiddenPin').value = currentPin;
                
                // Clear PIN from memory
                currentPin = '';
                clearPin();
                
                // Submit form
                setTimeout(() => {
                    document.getElementById('pinForm').submit();
                }, 500);
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
                window.location.href = '<?= htmlspecialchars($payment_data['cancel_url']) ?>?status=cancelled&order_id=<?= urlencode($payment_data['order_id']) ?>';
            }
        }
        
        // Handle successful PHP response
        <?php if ($success): ?>
        setTimeout(() => {
            window.location.href = 'payment_process.php';
        }, 3000);
        <?php endif; ?>
        
        // Prevent form submission on Enter key in PIN inputs
        document.querySelectorAll('.pin-digit').forEach(input => {
            input.addEventListener('keydown', function(e) {
                e.preventDefault();
            });
        });
        
        // Security: Clear PIN from memory on page unload
        window.addEventListener('beforeunload', function() {
            currentPin = '';
            clearPin();
        });
        
        // Auto-focus management (for accessibility)
        window.addEventListener('load', function() {
            // Focus first empty PIN input
            const firstEmpty = document.querySelector('.pin-digit:not(.filled)');
            if (firstEmpty) {
                firstEmpty.focus();
            }
        });
    </script>
</body>
</html>