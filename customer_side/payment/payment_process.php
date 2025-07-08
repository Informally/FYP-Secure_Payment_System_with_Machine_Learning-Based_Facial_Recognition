<?php
// customer_side/payment/payment_process.php - Complete Payment Processing with Gateway Redirects
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

// Auto-process payment when page loads (since both verifications are complete)
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
        SELECT id, merchant_id, name, is_active, return_url 
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
    $conn->close();
    
    // ✅ NEW: Redirect to gateway success page with merchant URL
    $success_url = "payment_success.php?" . http_build_query([
        'order_id' => $payment_data['order_id'],
        'transaction_id' => $transaction_id,
        'amount' => $payment_data['amount'],
        'currency' => $payment_data['currency'],
        'merchant_name' => $merchant['name'],
        'merchant_url' => $merchant['return_url'] // Pass merchant main page URL
    ]);
    
    header("Location: " . $success_url);
    exit();
    
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
    
    // Get merchant details for failed redirect
    $merchant_url = '';
    if (isset($conn)) {
        try {
            $stmt = $conn->prepare("SELECT return_url FROM merchants WHERE merchant_id = ?");
            $stmt->bind_param("s", $payment_data['merchant_id']);
            $stmt->execute();
            $result = $stmt->get_result();
            $merchant_data = $result->fetch_assoc();
            $merchant_url = $merchant_data['return_url'] ?? '';
            $stmt->close();
        } catch (Exception $fetch_error) {
            // Continue with empty merchant_url
        }
    }
    
    // ✅ NEW: Redirect to gateway failed page with merchant URL
    $failed_url = "payment_failed.php?" . http_build_query([
        'order_id' => $payment_data['order_id'],
        'amount' => $payment_data['amount'],
        'currency' => $payment_data['currency'],
        'merchant_name' => $payment_data['merchant_name'] ?? 'Unknown',
        'merchant_url' => $merchant_url,
        'error' => $e->getMessage()
    ]);
    
    header("Location: " . $failed_url);
    exit();
}
?>