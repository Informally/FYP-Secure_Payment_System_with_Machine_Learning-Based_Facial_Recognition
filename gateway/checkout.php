<?php
// gateway/checkout.php
// This is where your API key gets validated

session_start();
require_once '../customer_side/config.php';

// Validate merchant request
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    die('Invalid request method');
}

$merchant_id = $_POST['merchant_id'] ?? '';
$api_key = $_POST['api_key'] ?? '';  // ✅ GET THE API KEY
$amount = floatval($_POST['amount'] ?? 0);
$order_id = $_POST['order_id'] ?? '';
$currency = $_POST['currency'] ?? 'MYR';
$return_url = $_POST['return_url'] ?? '';
$cancel_url = $_POST['cancel_url'] ?? '';
$description = $_POST['description'] ?? '';

// ✅ STRICT VALIDATION: Check for missing credentials
if (empty($merchant_id)) {
    die('❌ Missing merchant ID');
}

if (empty($api_key)) {
    die('❌ Missing API key - unauthorized access');
}

// ✅ VALIDATE: Both merchant ID AND API key must match
$conn = dbConnect();
$merchant_stmt = $conn->prepare("
    SELECT id, name, api_key, is_active 
    FROM merchants 
    WHERE merchant_id = ? AND api_key = ?
");
$merchant_stmt->bind_param("ss", $merchant_id, $api_key);
$merchant_stmt->execute();
$merchant_result = $merchant_stmt->get_result();

if ($merchant_result->num_rows === 0) {
    // ✅ SECURITY: Don't reveal which credential was wrong
    die('❌ Invalid merchant credentials - unauthorized access');
}

$merchant = $merchant_result->fetch_assoc();

// ✅ Check if merchant is active
if (!$merchant['is_active']) {
    die('❌ Merchant account is not active');
}

// ✅ SECURITY LOG: Record successful authentication
error_log("Gateway: Successful authentication for merchant: {$merchant_id}");

// Validate payment amount
if ($amount <= 0 || $amount > 10000) {
    die('❌ Invalid payment amount: RM ' . $amount);
}

// ✅ Only create session AFTER successful validation
$_SESSION['gateway_payment'] = [
    'merchant_id' => $merchant_id,
    'merchant_name' => $merchant['name'],
    'amount' => $amount,
    'order_id' => $order_id,
    'currency' => $currency,
    'return_url' => $return_url,
    'cancel_url' => $cancel_url,
    'description' => $description,
    'timestamp' => time(),
    'validated' => true  // ✅ Mark as properly validated
];

$conn->close();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FacePay Payment Gateway</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea, #764ba2);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .gateway-container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 500px;
            width: 90%;
            text-align: center;
        }

        .gateway-header {
            margin-bottom: 30px;
        }

        .gateway-logo {
            font-size: 48px;
            color: #667eea;
            margin-bottom: 15px;
        }

        .gateway-title {
            font-size: 28px;
            color: #2c3e50;
            margin-bottom: 10px;
            font-weight: bold;
        }

        .gateway-subtitle {
            color: #7f8c8d;
            font-size: 16px;
        }

        .payment-details {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            margin: 30px 0;
            border: 2px solid #e9ecef;
        }

        .detail-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            font-size: 16px;
        }

        .detail-row:last-child {
            margin-bottom: 0;
            font-size: 20px;
            font-weight: bold;
            color: #e74c3c;
            border-top: 2px solid #dee2e6;
            padding-top: 15px;
        }

        .detail-label {
            color: #495057;
        }

        .detail-value {
            font-weight: bold;
            color: #2c3e50;
        }

        .merchant-info {
            background: linear-gradient(135deg, #e3f2fd, #f3e5f5);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            border: 2px solid #2196f3;
        }

        .action-buttons {
            display: flex;
            gap: 15px;
            margin-top: 30px;
        }

        .btn {
            flex: 1;
            padding: 15px 20px;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .btn-primary {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }

        .btn-secondary {
            background: #95a5a6;
            color: white;
        }

        .btn-secondary:hover {
            background: #7f8c8d;
            transform: translateY(-2px);
        }

        .security-badge {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
            color: #27ae60;
            font-size: 14px;
            font-weight: bold;
        }

        .auth-success {
            background: #e8f5e9;
            border: 2px solid #4caf50;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            font-size: 12px;
            color: #2e7d32;
        }
    </style>
</head>
<body>
    <div class="gateway-container">
        <div class="gateway-header">
            <div class="gateway-logo">
                <i class="fas fa-shield-alt"></i>
            </div>
            <h1 class="gateway-title">FacePay Gateway</h1>
            <p class="gateway-subtitle">Secure Facial Recognition Payment</p>
        </div>

        <div class="merchant-info">
            <h3><i class="fas fa-store"></i> Payment to <?= htmlspecialchars($merchant['name']) ?></h3>
            <p>Merchant ID: <?= htmlspecialchars($merchant_id) ?></p>
        </div>

        <div class="payment-details">
            <div class="detail-row">
                <span class="detail-label">Order ID:</span>
                <span class="detail-value"><?= htmlspecialchars($order_id) ?></span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Description:</span>
                <span class="detail-value"><?= htmlspecialchars($description) ?></span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Currency:</span>
                <span class="detail-value"><?= htmlspecialchars($currency) ?></span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Amount:</span>
                <span class="detail-value"><?= $currency ?> <?= number_format($amount, 2) ?></span>
            </div>
        </div>

        <div class="action-buttons">
            <button class="btn btn-secondary" onclick="cancelPayment()">
                <i class="fas fa-times"></i> Cancel
            </button>
            <button class="btn btn-primary" onclick="startFacialPayment()">
                <i class="fas fa-user-check"></i> Pay with Face
            </button>
        </div>

        <div class="security-badge">
            <i class="fas fa-lock"></i>
            <span>Secured by SSL & Biometric Authentication</span>
        </div>

        <div class="auth-success">
            ✅ <strong>Merchant Authenticated Successfully</strong><br>
            API credentials verified and session secured
        </div>
    </div>

    <script>
        function startFacialPayment() {
            // Redirect to facial recognition flow with validated session
            window.location.href = '../customer_side/payment/payment_scan.php';
        }

        function cancelPayment() {
            // Redirect back to merchant's cancel URL
            window.location.href = '<?= htmlspecialchars($cancel_url) ?>?status=cancelled&order_id=<?= urlencode($order_id) ?>';
        }

        // Auto-cancel after 10 minutes
        setTimeout(() => {
            alert('Payment session expired. Redirecting back to merchant.');
            cancelPayment();
        }, 600000);
    </script>
</body>
</html>