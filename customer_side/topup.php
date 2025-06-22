<?php
// topup.php - Complete Version with Fixed Database Issues
require_once 'config.php';
require_once 'encrypt.php';

// Check authentication
if (!SessionManager::start() || !SessionManager::requireAuthLevel('basic')) {
    header("Location: login.php");
    exit();
}

$user_id = $_SESSION['authenticatedUserId'];
$user_db_id = $_SESSION['user_db_id'];
$username = $_SESSION['username'];

$success = "";
$error = "";

try {
    $conn = dbConnect();
    
    // Get current wallet balance
    $stmt = $conn->prepare("
        SELECT u.full_name, u.email, w.balance_encrypted, w.balance 
        FROM users u
        LEFT JOIN wallets w ON u.id = w.user_id
        WHERE u.id = ?
    ");
    $stmt->bind_param("i", $user_db_id);
    $stmt->execute();
    $result = $stmt->get_result();
    $user = $result->fetch_assoc();
    $stmt->close();
    
    // Get current balance
    $current_balance = '0.00';
    if ($user['balance_encrypted']) {
        try {
            $current_balance = aes_decrypt($user['balance_encrypted']);
        } catch (Exception $e) {
            $current_balance = $user['balance'] ?? '0.00';
        }
    } elseif ($user['balance']) {
        $current_balance = $user['balance'];
    }
    $current_balance = number_format(floatval($current_balance), 2, '.', '');
    
} catch (Exception $e) {
    error_log("Topup page error: " . $e->getMessage());
    $error = "Unable to load wallet information.";
    $current_balance = '0.00';
}

// Handle payment processing
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $action = $_POST['action'] ?? '';
    
    if ($action === 'process_payment') {
        // CSRF validation
        if (!validateCSRFToken($_POST['csrf_token'] ?? '')) {
            $error = "Invalid request. Please try again.";
        } else {
            // Get form data
            $amount = floatval($_POST['amount'] ?? 0);
            $card_number = preg_replace('/\s+/', '', $_POST['card_number'] ?? '');
            $card_expiry = $_POST['card_expiry'] ?? '';
            $card_cvv = $_POST['card_cvv'] ?? '';
            $card_name = trim($_POST['card_name'] ?? '');
            
            // Simulate payment processing
            $payment_result = simulatePaymentProcessing($amount, $card_number, $card_expiry, $card_cvv, $card_name);
            
            if ($payment_result['success']) {
                // Start database transaction for atomic operations
                $conn->begin_transaction();
                
                try {
                    // Calculate new balance
                    $old_balance = floatval($current_balance);
                    $new_balance = $old_balance + $amount;
                    $encrypted_balance = aes_encrypt(number_format($new_balance, 2, '.', ''));
                    $balance_plain = number_format($new_balance, 2, '.', '');
                    
                    // Check if wallet exists first
                    $check_stmt = $conn->prepare("SELECT user_id FROM wallets WHERE user_id = ? FOR UPDATE");
                    $check_stmt->bind_param("i", $user_db_id);
                    $check_stmt->execute();
                    $check_result = $check_stmt->get_result();
                    $wallet_exists = $check_result->fetch_assoc();
                    $check_stmt->close();
                    
                    if ($wallet_exists) {
                        // Update existing wallet
                        $stmt = $conn->prepare("
                            UPDATE wallets 
                            SET balance_encrypted = ?, balance = ?, updated_at = NOW() 
                            WHERE user_id = ?
                        ");
                        $stmt->bind_param("sdi", $encrypted_balance, $balance_plain, $user_db_id);
                        if (!$stmt->execute()) {
                            throw new Exception("Failed to update wallet: " . $stmt->error);
                        }
                        $stmt->close();
                    } else {
                        // Insert new wallet record
                        $stmt = $conn->prepare("
                            INSERT INTO wallets (user_id, balance_encrypted, balance, created_at, updated_at) 
                            VALUES (?, ?, ?, NOW(), NOW())
                        ");
                        $stmt->bind_param("isd", $user_db_id, $encrypted_balance, $balance_plain);
                        if (!$stmt->execute()) {
                            throw new Exception("Failed to create wallet: " . $stmt->error);
                        }
                        $stmt->close();
                    }
                    
                    // Record transaction - Fixed to match your actual database schema
                    $dup_check = $conn->prepare("SELECT id FROM transactions WHERE transaction_id = ?");
                    $dup_check->bind_param("s", $payment_result['reference']);
                    $dup_check->execute();
                    $dup_result = $dup_check->get_result();
                    $duplicate_exists = $dup_result->fetch_assoc();
                    $dup_check->close();
                    
                    if (!$duplicate_exists) {
                        $stmt = $conn->prepare("
                            INSERT INTO transactions (user_id, transaction_id, amount, status, payment_method, timestamp, transaction_type) 
                            VALUES (?, ?, ?, 'success', 'card', NOW(), 'topup')
                        ");
                        $stmt->bind_param("isd", $user_db_id, $payment_result['reference'], $amount);
                        if (!$stmt->execute()) {
                            throw new Exception("Failed to record transaction: " . $stmt->error);
                        }
                        $stmt->close();
                    }
                    
                    // Commit transaction
                    $conn->commit();
                    
                    // Log security event
                    SecurityUtils::logSecurityEvent($user_id, 'wallet_topup', 'success', [
                        'amount' => $amount,
                        'reference' => $payment_result['reference'],
                        'card_last4' => substr($card_number, -4)
                    ]);
                    
                    $success = "Top-up successful! RM " . number_format($amount, 2) . " has been added to your wallet.";
                    $current_balance = number_format($new_balance, 2, '.', '');
                    
                } catch (Exception $e) {
                    // Rollback transaction on error
                    $conn->rollback();
                    error_log("Topup database error: " . $e->getMessage());
                    
                    // Check if the transaction actually went through despite the error
                    $verify_stmt = $conn->prepare("SELECT id FROM transactions WHERE transaction_id = ? AND status = 'success'");
                    $verify_stmt->bind_param("s", $payment_result['reference']);
                    $verify_stmt->execute();
                    $verify_result = $verify_stmt->get_result();
                    $transaction_exists = $verify_result->fetch_assoc();
                    $verify_stmt->close();
                    
                    if ($transaction_exists) {
                        // Transaction was already processed successfully
                        $success = "Top-up already processed successfully! Please check your wallet balance.";
                        // Refresh current balance
                        $stmt = $conn->prepare("SELECT balance_encrypted, balance FROM wallets WHERE user_id = ?");
                        $stmt->bind_param("i", $user_db_id);
                        $stmt->execute();
                        $result = $stmt->get_result();
                        $wallet_data = $result->fetch_assoc();
                        $stmt->close();
                        
                        if ($wallet_data['balance_encrypted']) {
                            try {
                                $current_balance = aes_decrypt($wallet_data['balance_encrypted']);
                            } catch (Exception $e) {
                                $current_balance = $wallet_data['balance'] ?? '0.00';
                            }
                        } else {
                            $current_balance = $wallet_data['balance'] ?? '0.00';
                        }
                        $current_balance = number_format(floatval($current_balance), 2, '.', '');
                    } else {
                        $error = "Payment processed but failed to update wallet. Please contact support with reference: " . $payment_result['reference'];
                    }
                }
            } else {
                $error = $payment_result['message'];
                SecurityUtils::logSecurityEvent($user_id, 'wallet_topup_failed', 'failure', [
                    'amount' => $amount,
                    'reason' => $payment_result['message'],
                    'card_last4' => substr($card_number, -4)
                ]);
            }
        }
    }
}

// Simulate payment processing with realistic rules
function simulatePaymentProcessing($amount, $card_number, $expiry, $cvv, $name) {
    // Basic validation
    if ($amount < 1 || $amount > 1000) {
        return ['success' => false, 'message' => 'Invalid amount. Please enter between RM 1.00 and RM 1,000.00'];
    }
    
    if (strlen($card_number) !== 16) {
        return ['success' => false, 'message' => 'Card number must be 16 digits'];
    }
    
    if (strlen($cvv) < 3 || strlen($cvv) > 4) {
        return ['success' => false, 'message' => 'Invalid CVV'];
    }
    
    if (empty($name)) {
        return ['success' => false, 'message' => 'Cardholder name is required'];
    }
    
    // Test cards that always fail
    if ($card_number === '4000000000000002') {
        return ['success' => false, 'message' => 'Your card was declined. Please try a different payment method.'];
    }
    
    if ($card_number === '4000000000000119') {
        return ['success' => false, 'message' => 'Insufficient funds. Please check your account balance.'];
    }
    
    if ($card_number === '4000000000000127') {
        return ['success' => false, 'message' => 'Your card has expired. Please use a different card.'];
    }
    
    // Amount-based rules for simulation
    $amount_cents = intval($amount * 100);
    $last_digit = $amount_cents % 10;
    
    if ($last_digit === 1) { // .x1 amounts
        return ['success' => false, 'message' => 'Transaction declined by your bank. Please contact your bank or try again later.'];
    }
    
    if ($last_digit === 9 && substr(number_format($amount, 2), -1) === '9') { // .x9 amounts
        return ['success' => false, 'message' => 'Network timeout. Please try again in a few moments.'];
    }
    
    // Card type validation - Accept any 16-digit number
    $first_digit = substr($card_number, 0, 1);
    if ($first_digit === '4') { // Visa
        $success_rate = 98; // 98% success for Visa
    } elseif ($first_digit === '5') { // Mastercard  
        $success_rate = 95; // 95% success for Mastercard
    } elseif ($first_digit === '3') { // Amex
        $success_rate = 90; // 90% success for Amex
    } else {
        // Accept any other starting digit but with lower success rate
        $success_rate = 85; // 85% success for other cards
    }
    
    // Random failure simulation
    if (rand(1, 100) > $success_rate) {
        $random_errors = [
            'Transaction declined. Please contact your bank.',
            'Card verification failed. Please check your details.',
            'Your bank has declined this transaction.',
            'Daily limit exceeded. Please try again tomorrow.',
            'Card security check failed. Please try again.'
        ];
        return ['success' => false, 'message' => $random_errors[array_rand($random_errors)]];
    }
    
    // Success case
    $reference = 'TP' . date('Ymd') . strtoupper(bin2hex(random_bytes(4)));
    return [
        'success' => true, 
        'message' => 'Payment successful',
        'reference' => $reference
    ];
}

$csrf_token = generateCSRFToken();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Top Up Wallet - Secure FacePay</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary: #3498db;
            --primary-light: #e3f2fd;
            --primary-dark: #2980b9;
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
            background-color: var(--bg-light);
            color: var(--text-dark);
            line-height: 1.6;
        }
        
        /* Header */
        .header {
            background: linear-gradient(135deg, var(--primary), #667eea);
            color: white;
            padding: 20px 0;
            box-shadow: 0 2px 10px var(--shadow);
        }
        
        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
            display: flex;
            align-items: center;
        }
        
        .back-btn {
            color: white;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 8px;
            transition: background-color 0.3s;
        }
        
        .back-btn:hover {
            background-color: rgba(255,255,255,0.1);
            text-decoration: none;
            color: white;
        }
        
        .page-title {
            font-size: 24px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
            margin-left: 300px;
        }
        
        /* Main Content */
        .main-content {
            max-width: 800px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        
        .topup-container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        /* Current Balance Section */
        .balance-section {
            background: linear-gradient(135deg, var(--primary), #667eea);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .balance-label {
            font-size: 16px;
            opacity: 0.9;
            margin-bottom: 10px;
        }
        
        .balance-amount {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 15px;
        }
        
        .balance-note {
            font-size: 14px;
            opacity: 0.8;
        }
        
        /* Form Section */
        .form-section {
            padding: 40px;
        }
        
        .section-title {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 25px;
            color: var(--text-dark);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        /* Amount Selection */
        .amount-section {
            margin-bottom: 35px;
        }
        
        .amount-presets {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .amount-preset {
            padding: 15px;
            border: 2px solid var(--border);
            border-radius: 12px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: white;
        }
        
        .amount-preset:hover {
            border-color: var(--primary);
            background: var(--primary-light);
        }
        
        .amount-preset.active {
            border-color: var(--primary);
            background: var(--primary);
            color: white;
        }
        
        .preset-amount {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .preset-label {
            font-size: 12px;
            opacity: 0.8;
        }
        
        .custom-amount {
            margin-top: 20px;
        }
        
        .amount-input {
            width: 100%;
            padding: 15px 20px;
            border: 2px solid var(--border);
            border-radius: 12px;
            font-size: 18px;
            font-weight: 600;
            text-align: center;
            color: var(--text-dark);
        }
        
        .amount-input:focus {
            border-color: var(--primary);
            outline: none;
        }
        
        /* Payment Form */
        .payment-form {
            margin-top: 35px;
            padding-top: 35px;
            border-top: 2px solid var(--border);
        }
        
        .form-row {
            display: flex;
            gap: 20px;
            margin-bottom: 25px;
        }
        
        .form-group {
            flex: 1;
        }
        
        .form-group.full-width {
            flex: none;
            width: 100%;
        }
        
        .form-group label {
            display: block;
            font-size: 14px;
            font-weight: 600;
            color: var(--text-medium);
            margin-bottom: 8px;
        }
        
        .form-input {
            width: 100%;
            padding: 15px;
            border: 2px solid var(--border);
            border-radius: 12px;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        
        .form-input:focus {
            border-color: var(--primary);
            outline: none;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }
        
        .card-input {
            position: relative;
        }
        
        .card-icon {
            position: absolute;
            right: 15px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 24px;
            color: var(--text-light);
        }
        
        /* Security Badges */
        .security-badges {
            display: flex;
            align-items: center;
            gap: 15px;
            margin: 25px 0;
            padding: 20px;
            background: var(--bg-light);
            border-radius: 12px;
        }
        
        .security-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: var(--text-medium);
        }
        
        .security-badge i {
            color: var(--success);
        }
        
        /* Payment Button */
        .payment-button {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, var(--success), #27ae60);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .payment-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(56, 161, 105, 0.3);
        }
        
        .payment-button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        /* Messages */
        .message {
            padding: 15px 18px;
            border-radius: 12px;
            margin-bottom: 25px;
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
        
        /* Loading Animation */
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid transparent;
            border-top: 2px solid currentColor;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Test Card Info */
        .test-card-info {
            background: #2d3748;
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-size: 12px;
            margin-top: 15px;
            line-height: 1.4;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .main-content {
                padding: 20px 15px;
            }
            
            .form-section {
                padding: 25px 20px;
            }
            
            .form-row {
                flex-direction: column;
                gap: 15px;
            }
            
            .amount-presets {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .page-title {
                margin-left: 0;
                margin-top: 15px;
            }
            
            .header-content {
                flex-direction: column;
                align-items: flex-start;
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="header-content">
            <a href="dashboard.php" class="back-btn">
                <i class="fas fa-arrow-left"></i>
                Back to Dashboard
            </a>
            <div class="page-title">
                <i class="fas fa-credit-card"></i>
                Top Up Wallet
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="main-content">
        <div class="topup-container">
            <!-- Current Balance -->
            <div class="balance-section">
                <div class="balance-label">Current Wallet Balance</div>
                <div class="balance-amount">RM <?= $current_balance ?></div>
                <div class="balance-note">Secure and encrypted wallet balance</div>
            </div>
            
            <!-- Form Section -->
            <div class="form-section">
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
                
                <form method="POST" id="topupForm">
                    <input type="hidden" name="csrf_token" value="<?= $csrf_token ?>">
                    <input type="hidden" name="action" value="process_payment">
                    
                    <!-- Amount Selection -->
                    <div class="amount-section">
                        <h3 class="section-title">
                            <i class="fas fa-money-bill-wave"></i>
                            Select Amount
                        </h3>
                        
                        <div class="amount-presets">
                            <div class="amount-preset" data-amount="10">
                                <div class="preset-amount">RM 10</div>
                                <div class="preset-label">Quick top-up</div>
                            </div>
                            <div class="amount-preset" data-amount="20">
                                <div class="preset-amount">RM 20</div>
                                <div class="preset-label">Most popular</div>
                            </div>
                            <div class="amount-preset" data-amount="50">
                                <div class="preset-amount">RM 50</div>
                                <div class="preset-label">Great value</div>
                            </div>
                            <div class="amount-preset" data-amount="100">
                                <div class="preset-amount">RM 100</div>
                                <div class="preset-label">Maximum</div>
                            </div>
                        </div>
                        
                        <div class="custom-amount">
                            <input type="number" 
                                   name="amount" 
                                   id="amountInput" 
                                   class="amount-input" 
                                   placeholder="Enter custom amount" 
                                   min="1" 
                                   max="1000" 
                                   step="0.01" 
                                   required>
                        </div>
                    </div>
                    
                    <!-- Payment Form -->
                    <div class="payment-form">
                        <h3 class="section-title">
                            <i class="fas fa-credit-card"></i>
                            Payment Details
                        </h3>
                        
                        <div class="form-group full-width">
                            <label for="cardNumber">Card Number</label>
                            <div class="card-input">
                                <input type="text" 
                                       id="cardNumber" 
                                       name="card_number" 
                                       class="form-input" 
                                       placeholder="1234 5678 9012 3456" 
                                       maxlength="19" 
                                       required>
                                <i class="fas fa-credit-card card-icon" id="cardIcon"></i>
                            </div>
                        </div>
                        
                        <div class="form-row">
                            <div class="form-group">
                                <label for="cardExpiry">Expiry Date</label>
                                <input type="text" 
                                       id="cardExpiry" 
                                       name="card_expiry" 
                                       class="form-input" 
                                       placeholder="MM/YY" 
                                       maxlength="5" 
                                       required>
                            </div>
                            <div class="form-group">
                                <label for="cardCvv">CVV</label>
                                <input type="text" 
                                       id="cardCvv" 
                                       name="card_cvv" 
                                       class="form-input" 
                                       placeholder="123" 
                                       maxlength="4" 
                                       required>
                            </div>
                        </div>
                        
                        <div class="form-group full-width">
                            <label for="cardName">Cardholder Name</label>
                            <input type="text" 
                                   id="cardName" 
                                   name="card_name" 
                                   class="form-input" 
                                   placeholder="John Doe" 
                                   value="<?= htmlspecialchars($user['full_name'] ?? '') ?>"
                                   required>
                        </div>
                        
                        <!-- Test Card Information -->
                        <div class="test-card-info">
                            <strong>💡 Test Cards for Simulation:</strong><br>
                            • <strong>4242 4242 4242 4242</strong> - Always succeeds<br>
                            • <strong>4000 0000 0000 0002</strong> - Always declined<br>
                            • <strong>4000 0000 0000 0119</strong> - Insufficient funds<br>
                            • <strong>Any other 16-digit number</strong> - Random results based on amount
                        </div>
                        
                        <!-- Submit Button -->
                        <br><button type="submit" class="payment-button" id="payButton">
                            <i class="fas fa-credit-card"></i>
                            <span id="payButtonText">Pay Now</span>
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </main>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('topupForm');
            const amountInput = document.getElementById('amountInput');
            const presets = document.querySelectorAll('.amount-preset');
            const cardNumberInput = document.getElementById('cardNumber');
            const cardExpiryInput = document.getElementById('cardExpiry');
            const cardCvvInput = document.getElementById('cardCvv');
            const cardIcon = document.getElementById('cardIcon');
            const payButton = document.getElementById('payButton');
            const payButtonText = document.getElementById('payButtonText');
            
            // Amount preset selection
            presets.forEach(preset => {
                preset.addEventListener('click', function() {
                    const amount = this.dataset.amount;
                    amountInput.value = amount;
                    
                    // Update active state
                    presets.forEach(p => p.classList.remove('active'));
                    this.classList.add('active');
                });
            });
            
            // Clear preset selection when typing custom amount
            amountInput.addEventListener('input', function() {
                presets.forEach(p => p.classList.remove('active'));
                
                // Validate amount
                const amount = parseFloat(this.value);
                if (amount > 1000) {
                    this.value = '1000';
                } else if (amount < 0) {
                    this.value = '';
                }
            });
            
            // Card number formatting and validation
            cardNumberInput.addEventListener('input', function() {
                let value = this.value.replace(/\s+/g, '').replace(/[^0-9]/gi, '');
                let formattedValue = value.match(/.{1,4}/g)?.join(' ') || value;
                
                // Limit to 16 digits
                if (value.length > 16) {
                    value = value.substring(0, 16);
                    formattedValue = value.match(/.{1,4}/g)?.join(' ') || value;
                }
                
                this.value = formattedValue;
                
                // Update card icon based on first digit
                const firstDigit = value.charAt(0);
                if (firstDigit === '4') {
                    cardIcon.className = 'fab fa-cc-visa card-icon';
                    cardIcon.style.color = '#1a1f71';
                } else if (firstDigit === '5') {
                    cardIcon.className = 'fab fa-cc-mastercard card-icon';
                    cardIcon.style.color = '#eb001b';
                } else if (firstDigit === '3') {
                    cardIcon.className = 'fab fa-cc-amex card-icon';
                    cardIcon.style.color = '#006fcf';
                } else if (value.length > 0) {
                    cardIcon.className = 'fas fa-credit-card card-icon';
                    cardIcon.style.color = '#4a5568';
                } else {
                    cardIcon.className = 'fas fa-credit-card card-icon';
                    cardIcon.style.color = '#718096';
                }
            });
            
            // Card expiry formatting (MM/YY)
            cardExpiryInput.addEventListener('input', function() {
                let value = this.value.replace(/\D/g, '');
                if (value.length >= 2) {
                    value = value.substring(0, 2) + '/' + value.substring(2, 4);
                }
                this.value = value;
            });
            
            // CVV validation - only numbers
            cardCvvInput.addEventListener('input', function() {
                this.value = this.value.replace(/\D/g, '');
            });
            
            // Form submission with loading state
            form.addEventListener('submit', function() {
                payButton.disabled = true;
                payButtonText.innerHTML = '<div class="spinner"></div> Processing...';
                
                // Re-enable button after 30 seconds as fallback
                setTimeout(function() {
                    payButton.disabled = false;
                    payButtonText.innerHTML = 'Pay Now';
                }, 30000);
            });
            
            // Real-time form validation
            function validateForm() {
                const amount = parseFloat(amountInput.value);
                const cardNumber = cardNumberInput.value.replace(/\s/g, '');
                const expiry = cardExpiryInput.value;
                const cvv = cardCvvInput.value;
                const name = document.getElementById('cardName').value.trim();
                
                const isValid = amount >= 1 && amount <= 1000 &&
                               cardNumber.length === 16 &&
                               expiry.length === 5 &&
                               cvv.length >= 3 &&
                               name.length > 0;
                
                payButton.disabled = !isValid;
                return isValid;
            }
            
            // Add validation listeners
            [amountInput, cardNumberInput, cardExpiryInput, cardCvvInput, document.getElementById('cardName')]
                .forEach(input => {
                    input.addEventListener('input', validateForm);
                    input.addEventListener('blur', validateForm);
                });
            
            // Initial validation
            validateForm();
            
            // Prevent form submission with Enter key on input fields (except submit button)
            document.querySelectorAll('input').forEach(input => {
                input.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        const isValid = validateForm();
                        if (isValid) {
                            form.submit();
                        }
                    }
                });
            });
            
            // Auto-focus next field functionality
            cardNumberInput.addEventListener('input', function() {
                if (this.value.replace(/\s/g, '').length === 16) {
                    cardExpiryInput.focus();
                }
            });
            
            cardExpiryInput.addEventListener('input', function() {
                if (this.value.length === 5) {
                    cardCvvInput.focus();
                }
            });
            
            cardCvvInput.addEventListener('input', function() {
                if (this.value.length === 3 || this.value.length === 4) {
                    document.getElementById('cardName').focus();
                }
            });
        });
    </script>
</body>
</html>