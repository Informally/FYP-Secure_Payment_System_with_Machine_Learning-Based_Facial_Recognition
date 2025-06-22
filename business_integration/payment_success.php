<?php
// business_integration/payment_success.php
// Handle successful payments from FacePay Gateway

// Get return parameters from gateway
$status = $_GET['status'] ?? 'success';
$order_id = $_GET['order_id'] ?? 'Unknown';
$transaction_id = $_GET['transaction_id'] ?? 'Unknown';
$amount = $_GET['amount'] ?? '0.00';
$payment_id = $_GET['payment_id'] ?? $transaction_id;

// For demo purposes, validate this came from the gateway
if ($status !== 'success') {
    header("Location: payment_failed.php?error=Invalid payment status&order_id=" . urlencode($order_id));
    exit();
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Payment Successful - Prototype Kiosk</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .kiosk-header {
            background: linear-gradient(45deg, #27ae60, #2ecc71);
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

        .success-container {
            max-width: 600px;
            margin: 40px auto;
            padding: 0 20px;
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .success-card {
            background: white;
            border-radius: 20px;
            padding: 50px;
            box-shadow: 0 20px 40px rgba(39, 174, 96, 0.2);
            border: 4px solid #27ae60;
            text-align: center;
            width: 100%;
            animation: slideIn 0.6s ease-out;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .success-icon {
            font-size: 100px;
            color: #27ae60;
            margin-bottom: 30px;
            animation: bounce 1s ease-in-out;
        }

        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-10px); }
            60% { transform: translateY(-5px); }
        }

        .success-title {
            font-size: 32px;
            color: #27ae60;
            margin-bottom: 15px;
            font-weight: bold;
        }

        .success-message {
            font-size: 18px;
            color: #2c3e50;
            margin-bottom: 40px;
            line-height: 1.6;
        }

        .gateway-info {
            background: linear-gradient(135deg, #e3f2fd, #f3e5f5);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            border: 2px solid #2196f3;
            color: #1565c0;
        }

        .transaction-details {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 40px;
            border: 2px solid #e9ecef;
        }

        .detail-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding: 10px 0;
            border-bottom: 1px solid #dee2e6;
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

        .amount-highlight {
            color: #27ae60;
            font-size: 24px;
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

        .countdown {
            margin-top: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            border: 2px solid #f39c12;
            border-radius: 15px;
            color: #d68910;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="kiosk-header">
        <h1><i class="fas fa-check-circle"></i> Payment Successful</h1>
    </div>

    <div class="success-container">
        <div class="success-card">
            <div class="success-icon">
                <i class="fas fa-check-circle"></i>
            </div>
            
            <h2 class="success-title">Payment Completed!</h2>
            <p class="success-message">
                Your payment has been successfully processed through the FacePay Gateway.
                Thank you for using our kiosk service!
            </p>

            <div class="gateway-info">
                <strong><i class="fas fa-shield-alt"></i> Processed by FacePay Gateway</strong><br>
                Your payment was securely processed using our payment gateway integration.
                This demonstrates external merchant API integration.
            </div>

            <div class="transaction-details">
                <div class="detail-row">
                    <span class="detail-label">
                        <i class="fas fa-receipt"></i> Order ID:
                    </span>
                    <span class="detail-value"><?= htmlspecialchars($order_id) ?></span>
                </div>
                
                <div class="detail-row">
                    <span class="detail-label">
                        <i class="fas fa-credit-card"></i> Payment ID:
                    </span>
                    <span class="detail-value"><?= htmlspecialchars($payment_id) ?></span>
                </div>
                
                <div class="detail-row">
                    <span class="detail-label">
                        <i class="fas fa-money-bill-wave"></i> Amount Paid:
                    </span>
                    <span class="detail-value amount-highlight">
                        MYR <?= number_format(floatval($amount), 2) ?>
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
                        <i class="fas fa-user-shield"></i> Payment Method:
                    </span>
                    <span class="detail-value">FacePay Gateway (Facial Recognition)</span>
                </div>

                <div class="detail-row">
                    <span class="detail-label">
                        <i class="fas fa-code"></i> Integration Type:
                    </span>
                    <span class="detail-value">External Merchant API</span>
                </div>
            </div>

            <a href="e-commerce.php" class="btn btn-primary">
                <i class="fas fa-shopping-cart"></i> Continue Shopping
            </a>

            <div class="countdown" id="countdown">
                <i class="fas fa-clock"></i> 
                Automatically returning to main menu in <span id="timer">30</span> seconds...
            </div>
        </div>
    </div>

    <script>
        // Clear cart data since payment is complete
        sessionStorage.clear();
        localStorage.clear();
        
        // Countdown timer
        let timeLeft = 30;
        const timerElement = document.getElementById('timer');
        const countdownElement = document.getElementById('countdown');

        const countdown = setInterval(() => {
            timeLeft--;
            timerElement.textContent = timeLeft;
            
            if (timeLeft <= 10) {
                countdownElement.style.background = 'linear-gradient(135deg, #ffebee, #ffcdd2)';
                countdownElement.style.borderColor = '#f44336';
                countdownElement.style.color = '#c62828';
            }
            
            if (timeLeft <= 0) {
                clearInterval(countdown);
                window.location.href = 'e-commerce.php';
            }
        }, 1000);

        // Prevent back button navigation
        history.pushState(null, null, location.href);
        window.onpopstate = function () {
            history.go(1);
        };

        // Log the successful payment for demo purposes
        console.log('Gateway Success Response:', {
            status: '<?= addslashes($status) ?>',
            order_id: '<?= addslashes($order_id) ?>',
            transaction_id: '<?= addslashes($transaction_id) ?>',
            amount: '<?= addslashes($amount) ?>',
            timestamp: new Date().toISOString()
        });

        // Demo: Show confetti or celebration effect
        function showCelebration() {
            const colors = ['#27ae60', '#2ecc71', '#f39c12', '#e74c3c'];
            for (let i = 0; i < 50; i++) {
                setTimeout(() => {
                    const confetti = document.createElement('div');
                    confetti.style.position = 'fixed';
                    confetti.style.left = Math.random() * 100 + 'vw';
                    confetti.style.top = '-10px';
                    confetti.style.width = '10px';
                    confetti.style.height = '10px';
                    confetti.style.background = colors[Math.floor(Math.random() * colors.length)];
                    confetti.style.borderRadius = '50%';
                    confetti.style.pointerEvents = 'none';
                    confetti.style.zIndex = '9999';
                    confetti.style.animation = 'fall 3s linear forwards';
                    document.body.appendChild(confetti);
                    
                    setTimeout(() => confetti.remove(), 3000);
                }, i * 100);
            }
        }

        // Add CSS for confetti animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes fall {
                to {
                    transform: translateY(100vh) rotate(360deg);
                }
            }
        `;
        document.head.appendChild(style);

        // Show celebration after a short delay
        setTimeout(showCelebration, 1000);
    </script>
</body>
</html>