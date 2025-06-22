<?php
// business_integration/payment_failed.php
// Handle failed/cancelled payments from FacePay Gateway

// Get return parameters from gateway
$status = $_GET['status'] ?? 'failed';
$order_id = $_GET['order_id'] ?? 'Unknown';
$error_message = $_GET['error'] ?? 'Payment was cancelled or failed';
$amount = $_GET['amount'] ?? '0.00';

// For demo purposes, show the gateway return
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Payment Failed - Prototype Kiosk</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #ffebee, #ffcdd2);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .kiosk-header {
            background: linear-gradient(45deg, #e74c3c, #c0392b);
            padding: 15px 0;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        .kiosk-header h1 {
            color: white;
            font-size: 2.5rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .failed-container {
            max-width: 600px;
            margin: 40px auto;
            padding: 0 20px;
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .failed-card {
            background: white;
            border-radius: 20px;
            padding: 50px;
            box-shadow: 0 20px 40px rgba(231, 76, 60, 0.2);
            border: 4px solid #e74c3c;
            text-align: center;
            width: 100%;
            animation: slideIn 0.6s ease-out;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .failed-icon {
            font-size: 100px;
            color: #e74c3c;
            margin-bottom: 30px;
            animation: shake 0.8s ease-in-out;
        }

        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
            20%, 40%, 60%, 80% { transform: translateX(5px); }
        }

        .failed-title {
            font-size: 32px;
            color: #e74c3c;
            margin-bottom: 15px;
            font-weight: bold;
        }

        .failed-message {
            font-size: 18px;
            color: #2c3e50;
            margin-bottom: 40px;
            line-height: 1.6;
        }

        .gateway-info {
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            border: 2px solid #f39c12;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            color: #d68910;
        }

        .error-details {
            background: #fff5f5;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 40px;
            border: 2px solid #fed7d7;
        }

        .detail-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding: 10px 0;
            border-bottom: 1px solid #fed7d7;
        }

        .detail-row:last-child {
            border-bottom: none;
            margin-bottom: 0;
        }

        .detail-label {
            font-weight: bold;
            color: #495057;
            font-size: 16px;
        }

        .detail-value {
            font-weight: bold;
            color: #2c3e50;
            font-size: 16px;
        }

        .error-highlight {
            color: #e74c3c;
            font-size: 16px;
            background: #fed7d7;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #e74c3c;
        }

        .btn {
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            text-decoration: none;
            margin: 10px;
        }

        .btn-primary {
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
        }

        .btn-primary:hover {
            background: linear-gradient(45deg, #2980b9, #3498db);
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(52, 152, 219, 0.3);
        }

        .btn-warning {
            background: linear-gradient(45deg, #f39c12, #e67e22);
            color: white;
        }

        .btn-warning:hover {
            background: linear-gradient(45deg, #e67e22, #f39c12);
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(243, 156, 18, 0.3);
        }

        .countdown {
            margin-top: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #ffebee, #ffcdd2);
            border: 2px solid #e74c3c;
            border-radius: 15px;
            color: #c62828;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="kiosk-header">
        <h1><i class="fas fa-times-circle"></i> Payment Failed</h1>
    </div>

    <div class="failed-container">
        <div class="failed-card">
            <div class="failed-icon">
                <i class="fas fa-times-circle"></i>
            </div>
            
            <h2 class="failed-title">Payment Failed</h2>
            <p class="failed-message">
                Your payment could not be processed through the FacePay Gateway. 
                Please review the details below and try again.
            </p>

            <div class="gateway-info">
                <strong><i class="fas fa-exclamation-triangle"></i> Gateway Response</strong><br>
                This page was reached via FacePay Gateway redirect with cancellation/failure status.
            </div>

            <div class="error-details">
                <div class="detail-row">
                    <span class="detail-label">
                        <i class="fas fa-exclamation-triangle"></i> Status:
                    </span>
                    <span class="detail-value error-highlight">
                        <?= htmlspecialchars(ucfirst($status)) ?>
                    </span>
                </div>
                
                <div class="detail-row">
                    <span class="detail-label">
                        <i class="fas fa-receipt"></i> Order ID:
                    </span>
                    <span class="detail-value">
                        <?= htmlspecialchars($order_id) ?>
                    </span>
                </div>

                <?php if ($amount > 0): ?>
                <div class="detail-row">
                    <span class="detail-label">
                        <i class="fas fa-credit-card"></i> Attempted Amount:
                    </span>
                    <span class="detail-value">
                        RM <?= number_format(floatval($amount), 2) ?>
                    </span>
                </div>
                <?php endif; ?>
                
                <div class="detail-row">
                    <span class="detail-label">
                        <i class="fas fa-info-circle"></i> Reason:
                    </span>
                    <span class="detail-value error-highlight">
                        <?= htmlspecialchars($error_message) ?>
                    </span>
                </div>
                
                <div class="detail-row">
                    <span class="detail-label">
                        <i class="fas fa-calendar-alt"></i> Date & Time:
                    </span>
                    <span class="detail-value"><?= date('F j, Y - g:i A') ?></span>
                </div>

                <div class="detail-row">
                    <span class="detail-label">
                        <i class="fas fa-code"></i> Gateway Integration:
                    </span>
                    <span class="detail-value">FacePay API Response</span>
                </div>
            </div>

            <div>
                <a href="shopping_cart.php" class="btn btn-warning" onclick="preserveCartAndRetry()">
                    <i class="fas fa-redo"></i> Try Again
                </a>
                <a href="e-commerce.php" class="btn btn-primary" onclick="clearCartAndReturn()">
                    <i class="fas fa-shopping-cart"></i> Continue Shopping
                </a>
            </div>

            <div class="countdown" id="countdown">
                <i class="fas fa-clock"></i> 
                Automatically returning to main menu in <span id="timer">30</span> seconds...
            </div>
        </div>
    </div>

    <script>
        // DON'T clear cart data when payment fails - user might want to retry
        // sessionStorage.clear(); // REMOVED
        // localStorage.clear();   // REMOVED
        
        // Countdown timer
        let timeLeft = 30;
        const timerElement = document.getElementById('timer');
        const countdownElement = document.getElementById('countdown');

        const countdown = setInterval(() => {
            timeLeft--;
            timerElement.textContent = timeLeft;
            
            if (timeLeft <= 10) {
                countdownElement.style.background = 'linear-gradient(135deg, #ffebee, #ef5350)';
                countdownElement.style.color = 'white';
            }
            
            if (timeLeft <= 0) {
                clearInterval(countdown);
                clearCartAndReturn();
            }
        }, 1000);

        // Function to preserve cart and retry payment
        function preserveCartAndRetry() {
            // Keep the cart intact and go back to checkout
            return true; // Allow the link to proceed
        }

        // Function to clear cart and return to shopping
        function clearCartAndReturn() {
            // Clear cart when explicitly choosing to continue shopping
            sessionStorage.removeItem('kioskCart');
            localStorage.removeItem('kioskCart');
            window.location.href = 'e-commerce.php';
            return false; // Prevent default link behavior
        }

        // Log the gateway response for demo purposes
        console.log('Gateway Response:', {
            status: '<?= addslashes($status) ?>',
            order_id: '<?= addslashes($order_id) ?>',
            error: '<?= addslashes($error_message) ?>',
            amount: '<?= addslashes($amount) ?>',
            timestamp: new Date().toISOString()
        });
    </script>
</body>
</html>