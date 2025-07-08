<?php
// customer_side/payment/payment_failed.php - Gateway Failed Page
require_once '../config.php';

// Start session for cleanup
SessionManager::start();

// ✅ NEW: Check if data was passed via session (from PHP redirect)
if (isset($_SESSION['payment_redirect_data'])) {
    $payment_status = $_SESSION['payment_redirect_data']['status'];
    $order_id = $_SESSION['payment_redirect_data']['order_id'];
    $amount = $_SESSION['payment_redirect_data']['amount'];
    $currency = $_SESSION['payment_redirect_data']['currency'];
    $merchant_name = $_SESSION['payment_redirect_data']['merchant_name'];
    $merchant_url = $_SESSION['payment_redirect_data']['merchant_url'];
    $error = $_SESSION['payment_redirect_data']['error'];
    $status = $payment_status;
    
    // Clear the session data after using it
    unset($_SESSION['payment_redirect_data']);
} 
// ✅ Otherwise, check for POST/GET data (existing functionality)
else {
    $order_id = $_POST['order_id'] ?? $_GET['order_id'] ?? 'Unknown';
    $amount = $_POST['amount'] ?? $_GET['amount'] ?? '0.00';
    $currency = $_POST['currency'] ?? $_GET['currency'] ?? 'MYR';
    $merchant_name = $_POST['merchant_name'] ?? $_GET['merchant_name'] ?? 'Unknown Merchant';
    $merchant_url = $_POST['merchant_url'] ?? $_GET['merchant_url'] ?? '';
    $error = $_POST['error'] ?? $_GET['error'] ?? $_GET['reason'] ?? 'Payment was cancelled or failed';
    $status = $_POST['status'] ?? $_GET['status'] ?? 'failed';
}

// ✅ Store original payment data for retry BEFORE clearing session
$retry_data = [
    'merchant_id' => $_SESSION['gateway_payment']['merchant_id'] ?? 'MCD',
    'amount' => $amount,
    'currency' => $currency,
    'order_id' => $order_id,
    'description' => $_SESSION['gateway_payment']['description'] ?? 'Retry Payment'
];

// ✅ DECODE HTML ENTITIES to fix double encoding issues
$merchant_name = html_entity_decode($merchant_name, ENT_QUOTES, 'UTF-8');
$merchant_url = html_entity_decode($merchant_url, ENT_QUOTES, 'UTF-8');
$order_id = html_entity_decode($order_id, ENT_QUOTES, 'UTF-8');
$error = html_entity_decode($error, ENT_QUOTES, 'UTF-8');

// ✅ CRITICAL: Clear any remaining payment sessions
unset($_SESSION['gateway_payment']);
unset($_SESSION['payment_user']);

// Regenerate session ID for security
session_regenerate_id(true);
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Payment Failed - FacePay Gateway</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #e53e3e 0%, #f56565 100%);
            color: #2d3748;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        
        .failed-container {
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
        
        .failed-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 6px;
            background: linear-gradient(90deg, #e53e3e, #f56565);
        }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(50px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .failed-icon {
            width: 120px;
            height: 120px;
            background: #e53e3e;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 30px;
            animation: errorShake 1.2s ease-out;
        }
        
        @keyframes errorShake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-5px); }
            75% { transform: translateX(5px); }
        }
        
        .failed-icon i {
            font-size: 60px;
            color: white;
        }
        
        .gateway-branding {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px 25px;
            border-radius: 50px;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 25px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        h1 {
            color: #e53e3e;
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
            background: #fff5f5;
            border: 2px solid #fed7d7;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 40px;
        }
        
        .amount-failed {
            font-size: 48px;
            color: #e53e3e;
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
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 18px;
        }
        
        .error-details p {
            color: #4a5568;
            line-height: 1.6;
            margin-bottom: 10px;
        }
        
        .error-reason {
            background: #fed7d7;
            color: #e53e3e;
            padding: 12px 15px;
            border-radius: 8px;
            font-weight: 600;
            margin-top: 15px;
        }
        
        .redirect-info {
            background: #fffbeb;
            border: 2px solid #fbd38d;
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
            color: #ed8936;
            font-size: 20px;
            font-weight: 700;
        }
        
        .countdown-number {
            background: #ed8936;
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
        
        .manual-return {
            margin-top: 30px;
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .return-btn {
            background: #e53e3e;
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
            background: #c53030;
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(229, 62, 62, 0.3);
            text-decoration: none;
            color: white;
        }
        
        .retry-btn {
            background: #ed8936;
            color: white;
        }
        
        .retry-btn:hover {
            background: #d69e2e;
            box-shadow: 0 8px 20px rgba(237, 137, 54, 0.3);
        }
        
        .security-notice {
            margin-top: 25px;
            padding: 15px;
            background: #fff3cd;
            border: 1px solid #fbd38d;
            border-radius: 10px;
            font-size: 13px;
            color: #856404;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .failed-container {
                padding: 40px 25px;
            }
            
            h1 {
                font-size: 32px;
            }
            
            .amount-failed {
                font-size: 36px;
            }
            
            .merchant-name {
                font-size: 20px;
            }
            
            .manual-return {
                flex-direction: column;
                align-items: center;
            }
            
            .return-btn {
                width: 100%;
                max-width: 250px;
            }
        }
    </style>
</head>
<body>
    <div class="failed-container">
        <div class="gateway-branding">
            <i class="fas fa-shield-alt"></i>
            Secured by FacePay Gateway
        </div>
        
        <div class="failed-icon">
            <i class="fas fa-times"></i>
        </div>
        
        <h1>Payment <?= ucfirst($status) ?></h1>
        <p class="subtitle">
            <?= $status === 'cancelled' 
                ? 'Your payment has been cancelled' 
                : 'We encountered an issue processing your payment' ?>
        </p>
        
        <div class="payment-summary">
            <div class="amount-failed"><?= htmlspecialchars($currency) ?> <?= number_format(floatval($amount), 2) ?></div>
            <div class="merchant-name"><?= htmlspecialchars($merchant_name) ?></div>
            <div class="order-info">Order: <?= htmlspecialchars($order_id) ?></div>
        </div>
        
        <div class="error-details">
            <h4>
                <i class="fas fa-exclamation-triangle"></i>
                <?= $status === 'cancelled' ? 'Cancellation Details' : 'Error Details' ?>
            </h4>
            <p><strong>Reason:</strong></p>
            <div class="error-reason"><?= htmlspecialchars($error) ?></div>
            <p style="margin-top: 15px;">
                <strong>What to do:</strong> 
                <?= $status === 'cancelled' 
                    ? 'You can retry the payment or return to the merchant to continue shopping.' 
                    : 'Please check your account balance and try again. If the issue persists, contact support.' ?>
            </p>
        </div>
        
        <?php if (!empty($merchant_url)): ?>
        <div class="redirect-info">
            <div class="redirect-text">
                <i class="fas fa-info-circle"></i>
                Returning to <?= htmlspecialchars($merchant_name) ?> in
            </div>
            <div class="countdown">
                <span class="countdown-number" id="countdown">10</span>
                <span>seconds</span>
            </div>
        </div>
        <?php endif; ?>
        
        <div class="manual-return">
            <button class="return-btn retry-btn" onclick="retryPayment()">
                <i class="fas fa-redo"></i>
                Try Again
            </button>
            <button class="return-btn" onclick="returnToMerchant()">
                <i class="fas fa-arrow-left"></i>
                <?= !empty($merchant_url) ? 'Return to ' . htmlspecialchars($merchant_name) : 'Continue' ?>
            </button>
        </div>
        
        <div class="security-notice">
            <i class="fas fa-lock"></i>
            <strong>Secure Session:</strong> Your payment session has been safely cleared. You can start a new payment securely.
        </div>
    </div>

    <!-- ✅ FIXED: Hidden form for proper retry functionality -->
    <form id="retryGatewayForm" method="POST" action="../../gateway/checkout.php" style="display: none;">
        <input type="hidden" name="merchant_id" value="<?= htmlspecialchars($retry_data['merchant_id']) ?>">
        <input type="hidden" name="api_key" value="sk_1f55be80af366a8c3e87f53b14962a4f2ec4bbe14755af39">
        <input type="hidden" name="amount" value="<?= floatval($retry_data['amount']) ?>">
        <input type="hidden" name="order_id" value="<?= htmlspecialchars($retry_data['order_id']) ?>">
        <input type="hidden" name="currency" value="<?= htmlspecialchars($retry_data['currency']) ?>">
        <input type="hidden" name="description" value="<?= htmlspecialchars($retry_data['description']) ?>">
        <input type="hidden" name="return_url" value="http://localhost/FYP/business_integration/payment_success.php">
        <input type="hidden" name="cancel_url" value="http://localhost/FYP/business_integration/payment_failed.php">
    </form>

    <script>
        // Auto redirect countdown (longer for failed payments)
        <?php if (!empty($merchant_url)): ?>
        let countdown = 10;
        const countdownElement = document.getElementById('countdown');
        
        const timer = setInterval(() => {
            countdown--;
            if (countdownElement) {
                countdownElement.textContent = countdown;
            }
            
            if (countdown <= 0) {
                clearInterval(timer);
                returnToMerchant();
            }
        }, 1000);
        <?php endif; ?>
        
        // ✅ FIXED: Proper retry payment function
        function retryPayment() {
            // Clear any remaining client data
            if (typeof(Storage) !== "undefined") {
                sessionStorage.clear();
                localStorage.clear();
            }
            
            // ✅ CORRECT: Submit gateway form with proper POST data (not redirect to payment_scan.php)
            document.getElementById('retryGatewayForm').submit();
        }
        
        // ✅ FIXED: Clean merchant return function
        function returnToMerchant() {
            <?php if (!empty($merchant_url)): ?>
                // ✅ CLEAN: Just go to merchant main page without any parameters
                let merchantUrl = '<?= addslashes($merchant_url) ?>';
                window.location.href = merchantUrl; // CLEAN URL!
            <?php else: ?>
                // Fallback: redirect to gateway homepage
                window.location.href = '../../gateway/checkout.php';
            <?php endif; ?>
        }
        
        // ✅ SECURITY: Prevent browser back button
        history.pushState(null, null, location.href);
        window.onpopstate = function () {
            history.go(1);
        };
        
        // ✅ SECURITY: Clear client-side data
        window.addEventListener('beforeunload', function() {
            if (typeof(Storage) !== "undefined") {
                sessionStorage.clear();
                localStorage.removeItem('payment_session');
            }
        });
        
        // ✅ SECURITY: Disable right-click and developer tools
        document.addEventListener('contextmenu', function(e) {
            e.preventDefault();
        });
        
        document.addEventListener('keydown', function(e) {
            if (e.keyCode == 123 || 
                (e.ctrlKey && e.shiftKey && e.keyCode == 73) ||
                (e.ctrlKey && e.shiftKey && e.keyCode == 74) ||
                (e.ctrlKey && e.keyCode == 85)) {
                e.preventDefault();
                return false;
            }
        });
    </script>
</body>
</html>