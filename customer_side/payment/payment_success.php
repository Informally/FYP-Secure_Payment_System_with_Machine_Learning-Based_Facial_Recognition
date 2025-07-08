<?php
// customer_side/payment/payment_success.php - Gateway Success Page
require_once '../config.php';

// Start session for cleanup
SessionManager::start();

// Get parameters from payment_process.php
$order_id = $_GET['order_id'] ?? 'Unknown';
$transaction_id = $_GET['transaction_id'] ?? 'Unknown';
$amount = $_GET['amount'] ?? '0.00';
$currency = $_GET['currency'] ?? 'MYR';
$merchant_name = $_GET['merchant_name'] ?? 'Unknown Merchant';
$merchant_url = $_GET['merchant_url'] ?? '';

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
    <title>Payment Successful - FacePay Gateway</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #38a169 0%, #48bb78 100%);
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
            background: linear-gradient(90deg, #38a169, #48bb78);
        }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(50px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .success-icon {
            width: 120px;
            height: 120px;
            background: #38a169;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 30px;
            animation: successPulse 1.2s ease-out;
        }
        
        @keyframes successPulse {
            0% { transform: scale(0.5); opacity: 0; }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); opacity: 1; }
        }
        
        .success-icon i {
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
            color: #38a169;
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
            background: #f0fff4;
            border: 2px solid #c6f6d5;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 40px;
        }
        
        .amount-paid {
            font-size: 48px;
            color: #38a169;
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
        
        .manual-return {
            margin-top: 30px;
        }
        
        .return-btn {
            background: #38a169;
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
            background: #2f855a;
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(56, 161, 105, 0.3);
            text-decoration: none;
            color: white;
        }
        
        .security-notice {
            margin-top: 25px;
            padding: 15px;
            background: #f0f8ff;
            border: 1px solid #bfdbfe;
            border-radius: 10px;
            font-size: 13px;
            color: #1e40af;
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
        <div class="gateway-branding">
            <i class="fas fa-shield-alt"></i>
            Secured by FacePay Gateway
        </div>
        
        <div class="success-icon">
            <i class="fas fa-check"></i>
        </div>
        
        <h1>Payment Successful!</h1>
        <p class="subtitle">Your payment has been processed successfully</p>
        
        <div class="payment-summary">
            <div class="amount-paid"><?= htmlspecialchars($currency) ?> <?= number_format(floatval($amount), 2) ?></div>
            <div class="merchant-name"><?= htmlspecialchars($merchant_name) ?></div>
            <div class="order-info">Order: <?= htmlspecialchars($order_id) ?></div>
        </div>
        
        <?php if (!empty($merchant_url)): ?>
        <div class="redirect-info">
            <div class="redirect-text">
                <i class="fas fa-info-circle"></i>
                Returning to <?= htmlspecialchars($merchant_name) ?> in
            </div>
            <div class="countdown">
                <span class="countdown-number" id="countdown">5</span>
                <span>seconds</span>
            </div>
        </div>
        <?php endif; ?>
        
        <div class="transaction-id">
            <label>Transaction Reference</label>
            <div class="transaction-id-value"><?= htmlspecialchars($transaction_id) ?></div>
        </div>
        
        <div class="manual-return">
            <button class="return-btn" onclick="returnToMerchant()">
                <i class="fas fa-arrow-left"></i>
                <?= !empty($merchant_url) ? 'Return to ' . htmlspecialchars($merchant_name) : 'Continue' ?>
            </button>
        </div>
        
        <div class="security-notice">
            <i class="fas fa-lock"></i>
            <strong>Secure Transaction:</strong> Your payment session has been safely completed and all session data cleared.
        </div>
    </div>

    <script>
        // Auto redirect countdown
        <?php if (!empty($merchant_url)): ?>
        let countdown = 5;
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
        
        function returnToMerchant() {
            <?php if (!empty($merchant_url)): ?>
                // Build success URL with parameters for merchant
                const params = new URLSearchParams();
                params.append('payment', 'success');
                params.append('order_id', '<?= urlencode($order_id) ?>');
                params.append('transaction_id', '<?= urlencode($transaction_id) ?>');
                params.append('amount', '<?= floatval($amount) ?>');
                
                const merchantUrl = '<?= htmlspecialchars($merchant_url) ?>';
                const separator = merchantUrl.includes('?') ? '&' : '?';
                window.location.href = merchantUrl + separator + params.toString();
            <?php else: ?>
                // Fallback: redirect to gateway homepage
                alert('Payment completed successfully!');
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