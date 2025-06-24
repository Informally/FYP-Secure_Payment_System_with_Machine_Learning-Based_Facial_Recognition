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
            'completed_at' => date('Y-m-d H:i:s'),
            'return_url' => $payment_data['return_url'] ?? ''
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
            'order_id' => $payment_data['order_id'],
            'cancel_url' => $payment_data['cancel_url'] ?? ''
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
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, 
                <?= $payment_result['status'] === 'success' ? '#38a169 0%, #48bb78 100%' : '#e53e3e 0%, #f56565 100%' ?>);
            color: #2d3748;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        
        .success-container {
            background: white;
            border-radius: 24px;
            padding: 60px 40px;
            box-shadow: 0 25px 50px rgba(0,0,0,0.2);
            text-align: center;
            max-width: 600px;
            width: 100%;
            animation: slideUp 0.8s ease-out;
            position: relative;
            overflow: hidden;
        }
        
        .success-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 6px;
            background: <?= $payment_result['status'] === 'success' ? '#38a169' : '#e53e3e' ?>;
        }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(50px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .success-icon {
            width: 120px;
            height: 120px;
            background: <?= $payment_result['status'] === 'success' ? '#38a169' : '#e53e3e' ?>;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 30px;
            animation: <?= $payment_result['status'] === 'success' ? 'successPulse' : 'errorShake' ?> 1.2s ease-out;
        }
        
        @keyframes successPulse {
            0% { transform: scale(0.5); opacity: 0; }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); opacity: 1; }
        }
        
        @keyframes errorShake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-5px); }
            75% { transform: translateX(5px); }
        }
        
        .success-icon i {
            font-size: 60px;
            color: white;
        }
        
        h1 {
            color: <?= $payment_result['status'] === 'success' ? '#38a169' : '#e53e3e' ?>;
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 15px;
        }
        
        .subtitle {
            color: #718096;
            font-size: 20px;
            margin-bottom: 40px;
            line-height: 1.4;
        }
        
        .payment-summary {
            background: <?= $payment_result['status'] === 'success' ? '#f0fff4' : '#fff5f5' ?>;
            border: 2px solid <?= $payment_result['status'] === 'success' ? '#c6f6d5' : '#fed7d7' ?>;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 40px;
        }
        
        .amount-paid {
            font-size: 48px;
            color: <?= $payment_result['status'] === 'success' ? '#38a169' : '#e53e3e' ?>;
            font-weight: 800;
            margin-bottom: 10px;
        }
        
        .merchant-name {
            font-size: 24px;
            color: #2d3748;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        .order-info {
            color: #718096;
            font-size: 16px;
        }
        
        <?php if ($payment_result['status'] === 'success'): ?>
        .redirect-info {
            background: #e6fffa;
            border: 2px solid #81e6d9;
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 30px;
        }
        
        .redirect-text {
            color: #2d3748;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
        }
        
        .countdown {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            color: #38a169;
            font-size: 20px;
            font-weight: 700;
        }
        
        .countdown-number {
            background: #38a169;
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: 800;
            animation: countdownPulse 1s infinite;
        }
        
        @keyframes countdownPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        <?php else: ?>
        .error-details {
            background: #fff5f5;
            border: 2px solid #fed7d7;
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 30px;
            text-align: left;
        }
        
        .error-details h4 {
            color: #e53e3e;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .error-details p {
            color: #4a5568;
            line-height: 1.6;
        }
        <?php endif; ?>
        
        .manual-return {
            margin-top: 30px;
        }
        
        .return-btn {
            background: <?= $payment_result['status'] === 'success' ? '#38a169' : '#e53e3e' ?>;
            color: white;
            border: none;
            padding: 18px 40px;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
        }
        
        .return-btn:hover {
            background: <?= $payment_result['status'] === 'success' ? '#2f855a' : '#c53030' ?>;
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(<?= $payment_result['status'] === 'success' ? '56, 161, 105' : '229, 62, 62' ?>, 0.3);
            text-decoration: none;
            color: white;
        }
        
        .transaction-id {
            margin-top: 30px;
            padding: 20px;
            background: #f7fafc;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
        }
        
        .transaction-id label {
            display: block;
            font-size: 12px;
            color: #718096;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 8px;
            letter-spacing: 0.5px;
        }
        
        .transaction-id-value {
            font-family: 'Courier New', monospace;
            font-size: 16px;
            color: #2d3748;
            font-weight: 600;
            letter-spacing: 1px;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .success-container {
                padding: 40px 25px;
            }
            
            h1 {
                font-size: 32px;
            }
            
            .amount-paid {
                font-size: 36px;
            }
            
            .merchant-name {
                font-size: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="success-container">
        <div class="success-icon">
            <i class="fas fa-<?= $payment_result['status'] === 'success' ? 'check' : 'times' ?>"></i>
        </div>
        
        <h1><?= $payment_result['status'] === 'success' ? 'Payment Successful!' : 'Payment Failed' ?></h1>
        <p class="subtitle">
            <?= $payment_result['status'] === 'success' 
                ? 'Your payment has been processed successfully' 
                : 'We encountered an issue processing your payment' ?>
        </p>
        
        <div class="payment-summary">
            <div class="amount-paid"><?= $payment_result['currency'] ?> <?= number_format($payment_result['amount'], 2) ?></div>
            <div class="merchant-name"><?= htmlspecialchars($payment_result['merchant_name']) ?></div>
            <div class="order-info">Order: <?= htmlspecialchars($payment_result['order_id']) ?></div>
        </div>
        
        <?php if ($payment_result['status'] === 'success'): ?>
            <div class="redirect-info">
                <div class="redirect-text">
                    <i class="fas fa-info-circle"></i>
                    Returning to merchant in
                </div>
                <div class="countdown">
                    <span class="countdown-number" id="countdown">5</span>
                    <span>seconds</span>
                </div>
            </div>
            
            <div class="transaction-id">
                <label>Transaction Reference</label>
                <div class="transaction-id-value"><?= htmlspecialchars($payment_result['transaction_id']) ?></div>
            </div>
            
            <div class="manual-return">
                <button class="return-btn" onclick="returnToMerchant()">
                    <i class="fas fa-arrow-left"></i>
                    Return to <?= htmlspecialchars($payment_result['merchant_name']) ?>
                </button>
            </div>
        <?php else: ?>
            <div class="error-details">
                <h4>
                    <i class="fas fa-exclamation-triangle"></i>
                    Error Details
                </h4>
                <p><strong>Reason:</strong> <?= htmlspecialchars($payment_result['error']) ?></p>
                <p><strong>What to do:</strong> Please check your account balance and try again. If the issue persists, contact support.</p>
            </div>
            
            <div class="manual-return">
                <button class="return-btn" onclick="returnToMerchant()">
                    <i class="fas fa-redo"></i>
                    Try Again
                </button>
            </div>
        <?php endif; ?>
    </div>

    <script>
        // Auto redirect countdown (for success only)
        <?php if ($payment_result['status'] === 'success'): ?>
        let countdown = 5;
        const countdownElement = document.getElementById('countdown');
        
        const timer = setInterval(() => {
            countdown--;
            countdownElement.textContent = countdown;
            
            if (countdown <= 0) {
                clearInterval(timer);
                returnToMerchant();
            }
        }, 1000);
        <?php endif; ?>
        
        function returnToMerchant() {
            <?php if ($payment_result['status'] === 'success' && !empty($payment_result['return_url'])): ?>
                // Build success URL with parameters
                const params = new URLSearchParams();
                params.append('status', 'success');
                params.append('order_id', '<?= urlencode($payment_result['order_id']) ?>');
                params.append('transaction_id', '<?= urlencode($payment_result['transaction_id']) ?>');
                params.append('amount', '<?= $payment_result['amount'] ?>');
                params.append('payment_id', '<?= urlencode($payment_result['transaction_id']) ?>');
                
                const returnUrl = '<?= htmlspecialchars($payment_result['return_url']) ?>';
                const separator = returnUrl.includes('?') ? '&' : '?';
                window.location.href = returnUrl + separator + params.toString();
            <?php elseif ($payment_result['status'] === 'failed' && !empty($payment_result['cancel_url'])): ?>
                // Redirect to cancel URL for failed payments
                const params = new URLSearchParams();
                params.append('status', 'failed');
                params.append('order_id', '<?= urlencode($payment_result['order_id']) ?>');
                params.append('error', '<?= urlencode($payment_result['error']) ?>');
                params.append('amount', '<?= $payment_result['amount'] ?>');
                
                const cancelUrl = '<?= htmlspecialchars($payment_result['cancel_url']) ?>';
                const separator = cancelUrl.includes('?') ? '&' : '?';
                window.location.href = cancelUrl + separator + params.toString();
            <?php else: ?>
                // Fallback: redirect to gateway
                alert('Returning to payment gateway...');
                window.location.href = '../../gateway/checkout.php';
            <?php endif; ?>
        }
        
        // Prevent browser back button
        history.pushState(null, null, location.href);
        window.onpopstate = function () {
            history.go(1);
        };
        
        // Clean up on page unload
        window.addEventListener('beforeunload', function() {
            if (sessionStorage) {
                sessionStorage.clear();
            }
        });
    </script>
</body>
</html>