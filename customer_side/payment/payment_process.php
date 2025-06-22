<?php
// customer_side/payment/payment_process.php - Complete Payment Processing
require_once '../config.php';
require_once '../encrypt.php';

// Start session and check for payment session
SessionManager::start();

// Check if we have a valid payment session from gateway
if (!isset($_SESSION['gateway_payment'])) {
    header("Location: ../../gateway/checkout.php");
    exit();
}

// Check if both face and PIN verification were completed
if (!isset($_SESSION['payment_user']) || 
    !$_SESSION['payment_user']['face_verified'] || 
    !$_SESSION['payment_user']['pin_verified']) {
    header("Location: payment_scan.php");
    exit();
}

$payment_data = $_SESSION['gateway_payment'];
$payment_user = $_SESSION['payment_user'];
$error = "";
$success = "";
$payment_result = null;

// Auto-process payment when page loads (since both verifications are complete)
if ($_SERVER["REQUEST_METHOD"] == "GET") {
    try {
        $conn = dbConnect();
        
        // Start transaction
        $conn->begin_transaction();
        
        // Get user's current balance from wallets table and account details from users table
        $stmt = $conn->prepare("
            SELECT u.id, u.user_id, u.username, u.full_name, u.account_status,
                   w.balance_encrypted, w.balance 
            FROM users u
            LEFT JOIN wallets w ON u.id = w.user_id 
            WHERE u.id = ? AND u.account_status = 'active'
        ");
        $stmt->bind_param("i", $payment_user['id']);
        $stmt->execute();
        $result = $stmt->get_result();
        $user = $result->fetch_assoc();
        $stmt->close();
        
        if (!$user) {
            throw new Exception("User account not found or not active");
        }
        
        // Get current balance (same logic as topup.php)
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
        $current_balance = floatval($current_balance);
        
        // Check if user has sufficient balance
        if ($current_balance < $payment_data['amount']) {
            throw new Exception("Insufficient balance. Current balance: " . number_format($current_balance, 2) . " " . $payment_data['currency']);
        }
        
        // Generate unique transaction ID
        $transaction_id = 'TXN_' . time() . '_' . substr(md5(uniqid()), 0, 8);
        
        // Get merchant details - Updated for your table structure
        $stmt = $conn->prepare("
            SELECT id, merchant_id, name, is_active 
            FROM merchants 
            WHERE merchant_id = ? AND is_active = 1
        ");
        $stmt->bind_param("s", $payment_data['merchant_id']);
        $stmt->execute();
        $merchant_result = $stmt->get_result();
        $merchant = $merchant_result->fetch_assoc();
        $stmt->close();
        
        if (!$merchant) {
            throw new Exception("Merchant not found or not active");
        }
        
        // Deduct amount from user balance in wallets table (same logic as your working code)
        $new_balance = $current_balance - $payment_data['amount'];
        $encrypted_balance = aes_encrypt($new_balance);
        
        // Check if wallet exists and update (same approach as topup.php)
        $check_stmt = $conn->prepare("SELECT user_id FROM wallets WHERE user_id = ? FOR UPDATE");
        $check_stmt->bind_param("i", $payment_user['id']);
        $check_stmt->execute();
        $check_result = $check_stmt->get_result();
        $wallet_exists = $check_result->fetch_assoc();
        $check_stmt->close();
        
        if ($wallet_exists) {
            // Update existing wallet (matching your working pattern)
            $stmt = $conn->prepare("
                UPDATE wallets 
                SET balance = ?, balance_encrypted = ?, updated_at = NOW() 
                WHERE user_id = ?
            ");
            $stmt->bind_param("dsi", $new_balance, $encrypted_balance, $payment_user['id']);
            if (!$stmt->execute()) {
                throw new Exception("Failed to update wallet: " . $stmt->error);
            }
            $stmt->close();
        } else {
            // Create new wallet if doesn't exist
            $stmt = $conn->prepare("
                INSERT INTO wallets (user_id, balance, balance_encrypted, created_at, updated_at) 
                VALUES (?, ?, ?, NOW(), NOW())
            ");
            $stmt->bind_param("ids", $payment_user['id'], $new_balance, $encrypted_balance);
            if (!$stmt->execute()) {
                throw new Exception("Failed to create wallet: " . $stmt->error);
            }
            $stmt->close();
        }
        
        // Insert payment transaction into your transactions table
        // Check for duplicates first (like topup.php)
        $dup_check = $conn->prepare("SELECT id FROM transactions WHERE transaction_id = ?");
        $dup_check->bind_param("s", $transaction_id);
        $dup_check->execute();
        $dup_result = $dup_check->get_result();
        $duplicate_exists = $dup_result->fetch_assoc();
        $dup_check->close();
        
        if (!$duplicate_exists) {
            $stmt = $conn->prepare("
                INSERT INTO transactions (
                    user_id, transaction_id, amount, status, 
                    payment_method, order_id, merchant_id, merchant_name,
                    currency, transaction_type, timestamp
                ) VALUES (?, ?, ?, 'success', 'facial', ?, ?, ?, ?, 'payment', NOW())
            ");
            // Store as negative amount to indicate outgoing payment (like topup.php does for incoming)
            $negative_amount = -$payment_data['amount'];
            $stmt->bind_param(
                "isdssss", 
                $payment_user['id'], // Use user_id from payment_user session (integer ID)
                $transaction_id,
                $negative_amount,
                $payment_data['order_id'],
                $payment_data['merchant_id'],
                $merchant['name'],
                $payment_data['currency']
            );
            if (!$stmt->execute()) {
                throw new Exception("Failed to record transaction: " . $stmt->error);
            }
            $transaction_db_id = $conn->insert_id;
            $stmt->close();
        }
        
        // Log the successful payment
        SecurityUtils::logSecurityEvent($user['user_id'], 'payment_completed', 'success', [
            'transaction_id' => $transaction_id,
            'amount' => $payment_data['amount'],
            'merchant_id' => $payment_data['merchant_id'],
            'order_id' => $payment_data['order_id'],
            'new_balance' => $new_balance,
            'face_similarity' => $payment_user['face_similarity'],
            'liveness_score' => $payment_user['liveness_score']
        ]);
        
        // Commit transaction
        $conn->commit();
        
        // Prepare payment result
        $payment_result = [
            'status' => 'success',
            'transaction_id' => $transaction_id,
            'amount' => $payment_data['amount'],
            'currency' => $payment_data['currency'],
            'merchant_name' => $merchant['name'],
            'order_id' => $payment_data['order_id'],
            'user_balance' => $new_balance,
            'completed_at' => date('Y-m-d H:i:s')
        ];
        
        $success = "Payment completed successfully!";
        
        // Clear payment sessions
        unset($_SESSION['gateway_payment']);
        unset($_SESSION['payment_user']);
        
        $conn->close();
        
    } catch (Exception $e) {
        // Rollback transaction
        if (isset($conn)) {
            $conn->rollback();
            $conn->close();
        }
        
        error_log("Payment processing error: " . $e->getMessage());
        
        // Log the failed payment
        SecurityUtils::logSecurityEvent($payment_user['user_id'] ?? 'unknown', 'payment_failed', 'failure', [
            'error' => $e->getMessage(),
            'amount' => $payment_data['amount'],
            'merchant_id' => $payment_data['merchant_id'],
            'order_id' => $payment_data['order_id']
        ]);
        
        $error = $e->getMessage();
        
        // Create failed payment result
        $payment_result = [
            'status' => 'failed',
            'error' => $error,
            'amount' => $payment_data['amount'],
            'currency' => $payment_data['currency'],
            'merchant_name' => $payment_data['merchant_name'] ?? 'Unknown',
            'order_id' => $payment_data['order_id']
        ];
    }
}

$csrf_token = generateCSRFToken();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title><?= $payment_result['status'] === 'success' ? 'Payment Successful' : 'Payment Failed' ?> - FacePay</title>
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
            background: linear-gradient(135deg, 
                <?= $payment_result['status'] === 'success' ? '#38a169 0%, #48bb78 100%' : '#e53e3e 0%, #f56565 100%' ?>);
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
            max-width: 700px;
        }
        
        .result-container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
            text-align: center;
            animation: fadeUp 0.8s ease-out;
            position: relative;
            overflow: hidden;
        }
        
        .result-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: <?= $payment_result['status'] === 'success' ? 'var(--success)' : 'var(--error)' ?>;
        }
        
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-header {
            margin-bottom: 35px;
        }
        
        .result-icon {
            width: 100px;
            height: 100px;
            background: <?= $payment_result['status'] === 'success' ? 'var(--success)' : 'var(--error)' ?>;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 25px;
            animation: <?= $payment_result['status'] === 'success' ? 'successPulse' : 'errorShake' ?> 1s ease-out;
        }
        
        @keyframes successPulse {
            0% { transform: scale(0.8); opacity: 0; }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); opacity: 1; }
        }
        
        @keyframes errorShake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-5px); }
            75% { transform: translateX(5px); }
        }
        
        .result-icon i {
            font-size: 48px;
            color: white;
        }
        
        h2 {
            color: <?= $payment_result['status'] === 'success' ? 'var(--success)' : 'var(--error)' ?>;
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: var(--text-light);
            font-size: 18px;
            margin-bottom: 35px;
        }
        
        /* Transaction Details */
        .transaction-details {
            background: var(--bg-light);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            border: 2px solid <?= $payment_result['status'] === 'success' ? '#c6f6d5' : '#fed7d7' ?>;
            text-align: left;
        }
        
        .transaction-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
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
            margin-bottom: 8px;
            letter-spacing: 0.5px;
        }
        
        .detail-value {
            font-size: 16px;
            color: var(--text-dark);
            font-weight: 600;
            word-break: break-all;
        }
        
        .amount-value {
            font-size: 28px;
            color: <?= $payment_result['status'] === 'success' ? 'var(--success)' : 'var(--error)' ?>;
            font-weight: 700;
        }
        
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .status-success {
            background: #f0fff4;
            color: var(--success);
            border: 1px solid #c6f6d5;
        }
        
        .status-failed {
            background: #fff5f5;
            color: var(--error);
            border: 1px solid #fed7d7;
        }
        
        /* Error Message */
        .error-details {
            background: #fff5f5;
            border: 2px solid #fed7d7;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            text-align: left;
        }
        
        .error-details h4 {
            color: var(--error);
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .error-details p {
            color: var(--text-medium);
            line-height: 1.6;
        }
        
        /* Balance Update (Success only) */
        .balance-update {
            background: linear-gradient(135deg, #f0fff4 0%, #e6fffa 100%);
            border: 2px solid #c6f6d5;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
        }
        
        .balance-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .balance-item {
            text-align: center;
        }
        
        .balance-label {
            font-size: 12px;
            color: var(--text-light);
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .balance-amount {
            font-size: 20px;
            color: var(--success);
            font-weight: 700;
        }
        
        /* Security Summary */
        .security-summary {
            background: #f0f8ff;
            border: 1px solid #bfdbfe;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            text-align: left;
        }
        
        .security-summary h4 {
            color: var(--primary);
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .security-items {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .security-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }
        
        .security-item i {
            color: var(--success);
            font-size: 16px;
        }
        
        .security-text {
            font-size: 14px;
            color: var(--text-medium);
        }
        
        /* Action Buttons */
        .actions {
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
        
        .btn-primary {
            background: var(--primary);
            color: white;
        }
        
        .btn-primary:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
            text-decoration: none;
            color: white;
        }
        
        .btn-success {
            background: var(--success);
            color: white;
        }
        
        .btn-success:hover {
            background: #2f855a;
            transform: translateY(-2px);
            text-decoration: none;
            color: white;
        }
        
        .btn-secondary {
            background: white;
            color: var(--text-medium);
            border: 2px solid var(--border);
        }
        
        .btn-secondary:hover {
            background: var(--bg-light);
            border-color: var(--primary);
            text-decoration: none;
        }
        
        /* Print Receipt Button */
        .print-btn {
            background: #f8f9fa;
            color: var(--text-dark);
            border: 2px dashed var(--border);
        }
        
        .print-btn:hover {
            background: #e9ecef;
            border-style: solid;
            text-decoration: none;
        }
        
        /* Receipt Styles for Print */
        @media print {
            body {
                background: white !important;
                padding: 0;
                margin: 0;
            }
            
            .container {
                max-width: none;
                width: 100%;
            }
            
            .result-container {
                box-shadow: none;
                border: 1px solid #000;
                margin: 0;
                padding: 20px;
            }
            
            .actions, .btn {
                display: none !important;
            }
            
            .result-container::before {
                display: none;
            }
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .result-container {
                padding: 25px 20px;
            }
            
            .transaction-grid {
                grid-template-columns: 1fr;
                gap: 15px;
            }
            
            .balance-info {
                flex-direction: column;
            }
            
            .security-items {
                grid-template-columns: 1fr;
            }
            
            .actions {
                flex-direction: column;
                align-items: center;
            }
            
            .btn {
                width: 100%;
                max-width: 300px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="result-container">
            <div class="result-header">
                <div class="result-icon">
                    <i class="fas fa-<?= $payment_result['status'] === 'success' ? 'check' : 'times' ?>"></i>
                </div>
                <h2><?= $payment_result['status'] === 'success' ? 'Payment Successful!' : 'Payment Failed' ?></h2>
                <p class="subtitle">
                    <?= $payment_result['status'] === 'success' 
                        ? 'Your payment has been processed successfully' 
                        : 'We encountered an issue processing your payment' ?>
                </p>
            </div>
            
            <!-- Transaction Details -->
            <div class="transaction-details">
                <div class="transaction-grid">
                    <?php if ($payment_result['status'] === 'success'): ?>
                        <div class="detail-item">
                            <div class="detail-label">Transaction ID</div>
                            <div class="detail-value"><?= htmlspecialchars($payment_result['transaction_id']) ?></div>
                        </div>
                    <?php endif; ?>
                    
                    <div class="detail-item">
                        <div class="detail-label">Order ID</div>
                        <div class="detail-value"><?= htmlspecialchars($payment_result['order_id']) ?></div>
                    </div>
                    
                    <div class="detail-item">
                        <div class="detail-label">Merchant</div>
                        <div class="detail-value"><?= htmlspecialchars($payment_result['merchant_name']) ?></div>
                    </div>
                    
                    <div class="detail-item">
                        <div class="detail-label">Amount</div>
                        <div class="detail-value amount-value">
                            <?= $payment_result['currency'] ?> <?= number_format($payment_result['amount'], 2) ?>
                        </div>
                    </div>
                    
                    <div class="detail-item">
                        <div class="detail-label">Status</div>
                        <div class="detail-value">
                            <span class="status-badge status-<?= $payment_result['status'] === 'success' ? 'success' : 'failed' ?>">
                                <i class="fas fa-<?= $payment_result['status'] === 'success' ? 'check-circle' : 'times-circle' ?>"></i>
                                <?= ucfirst($payment_result['status']) ?>
                            </span>
                        </div>
                    </div>
                    
                    <?php if ($payment_result['status'] === 'success'): ?>
                        <div class="detail-item">
                            <div class="detail-label">Completed At</div>
                            <div class="detail-value"><?= htmlspecialchars($payment_result['completed_at']) ?></div>
                        </div>
                    <?php endif; ?>
                </div>
            </div>
            
            <?php if ($payment_result['status'] === 'failed'): ?>
                <!-- Error Details -->
                <div class="error-details">
                    <h4>
                        <i class="fas fa-exclamation-triangle"></i>
                        Error Details
                    </h4>
                    <p><strong>Reason:</strong> <?= htmlspecialchars($payment_result['error']) ?></p>
                    <p><strong>What to do:</strong> Please check your account balance and try again. If the issue persists, contact support.</p>
                </div>
            <?php else: ?>
                <!-- Balance Update (Success only) -->
                <!-- <div class="balance-update">
                    <div class="balance-info">
                        <div class="balance-item">
                            <div class="balance-label">Amount Paid</div>
                            <div class="balance-amount">-<?= $payment_result['currency'] ?> <?= number_format($payment_result['amount'], 2) ?></div>
                        </div>
                        <div class="balance-item">
                            <div class="balance-label">Remaining Balance</div>
                            <div class="balance-amount"><?= $payment_result['currency'] ?> <?= number_format($payment_result['user_balance'], 2) ?></div>
                        </div> -->
                    </div>
                </div>
                
                <!-- Security Summary -->
                <div class="security-summary">
                    <h4>
                        <i class="fas fa-shield-alt"></i>
                        Security Verification Completed
                    </h4>
                    <div class="security-items">
                        <div class="security-item">
                            <i class="fas fa-user-check"></i>
                            <span class="security-text">Face Recognition Verified</span>
                        </div>
                        <div class="security-item">
                            <i class="fas fa-eye"></i>
                            <span class="security-text">Liveness Detection Passed</span>
                        </div>
                        <div class="security-item">
                            <i class="fas fa-lock"></i>
                            <span class="security-text">PIN Authentication Successful</span>
                        </div>
                        <div class="security-item">
                            <i class="fas fa-encrypt"></i>
                            <span class="security-text">Transaction Encrypted</span>
                        </div>
                    </div>
                </div>
            <?php endif; ?>
            
            <!-- Action Buttons -->
            <div class="actions">
                <?php if ($payment_result['status'] === 'success'): ?>
                    <button class="btn print-btn" onclick="window.print()">
                        <i class="fas fa-print"></i>
                        Print Receipt
                    </button>
                    
                    <a href="../../dashboard.php" class="btn btn-success">
                        <i class="fas fa-home"></i>
                        Go to Dashboard
                    </a>
                    
                    <a href="../../transactions.php" class="btn btn-secondary">
                        <i class="fas fa-history"></i>
                        View Transactions
                    </a>
                <?php else: ?>
                    <a href="../../gateway/checkout.php" class="btn btn-primary">
                        <i class="fas fa-redo"></i>
                        Try Again
                    </a>
                    
                    <!-- <a href="../../dashboard.php" class="btn btn-secondary">
                        <i class="fas fa-home"></i>
                        Go to Dashboard
                    </a>
                    
                    <a href="../../support.php" class="btn btn-secondary">
                        <i class="fas fa-life-ring"></i>
                        Get Support
                    </a> -->
                <?php endif; ?>
            </div>
        </div>
    </div>

    <script>
        // Prevent back button after successful payment
        <?php if ($payment_result['status'] === 'success'): ?>
        history.pushState(null, null, location.href);
        window.onpopstate = function () {
            history.go(1);
        };
        <?php endif; ?>
        
        // Add some visual feedback on page load
        document.addEventListener('DOMContentLoaded', function() {
            // Add entrance animation delay for details
            const details = document.querySelector('.transaction-details');
            if (details) {
                details.style.opacity = '0';
                details.style.transform = 'translateY(20px)';
                setTimeout(() => {
                    details.style.transition = 'all 0.6s ease';
                    details.style.opacity = '1';
                    details.style.transform = 'translateY(0)';
                }, 500);
            }
        });
        
        // Security: Clear any sensitive data from browser
        window.addEventListener('beforeunload', function() {
            // Clear any cached form data
            if (sessionStorage) {
                sessionStorage.clear();
            }
        });
    </script>
</body>
</html>