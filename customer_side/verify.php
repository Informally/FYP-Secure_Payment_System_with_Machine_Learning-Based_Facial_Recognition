<?php
// Updated verify.php - Fixed for MySQLi and your database structure
require_once 'config.php';

$token = SecurityUtils::sanitizeInput($_GET['token'] ?? '');
$message = '';
$redirect = false;

if (!empty($token)) {
    try {
        $conn = dbConnect(); // MySQLi connection
        
        // Query matching your actual database structure
        $stmt = $conn->prepare("SELECT id, username, email FROM users WHERE verification_token = ? AND is_verified = 0");
        $stmt->bind_param("s", $token);
        $stmt->execute();
        $result = $stmt->get_result();
        $user = $result->fetch_assoc();

        if ($user) {
            // Mark user as verified (simplified to match your table structure)
            $update_stmt = $conn->prepare("UPDATE users SET is_verified = 1, verification_token = NULL WHERE id = ?");
            $update_stmt->bind_param("i", $user['id']);
            
            if ($update_stmt->execute()) {
                // Check if wallet already exists for this user
                $wallet_check = $conn->prepare("SELECT id FROM wallets WHERE user_id = ?");
                $wallet_check->bind_param("i", $user['id']);
                $wallet_check->execute();
                $wallet_result = $wallet_check->get_result();
                
                if ($wallet_result->num_rows == 0) {
                    // Create wallet for verified user (simple structure like your working register.php)
                    $wallet_stmt = $conn->prepare("INSERT INTO wallets (user_id, balance) VALUES (?, 50.00)");
                    $wallet_stmt->bind_param("i", $user['id']);
                    $wallet_stmt->execute();
                    
                    // Log welcome bonus transaction
                    $trans_stmt = $conn->prepare("INSERT INTO transactions (user_id, amount, status) VALUES (?, 50.00, 'success')");
                    $trans_stmt->bind_param("i", $user['id']);
                    $trans_stmt->execute();
                }
                
                // Log the verification
                SecurityUtils::logSecurityEvent($user['username'], 'email_verification', 'success', [
                    'email' => 'verified',
                    'welcome_bonus' => 50.00
                ]);
                
                // // Start session for user
                // SessionManager::start();
                // SessionManager::setAuthLevel('basic', $user['username']);
                // $_SESSION['username'] = $user['username'];
                // $_SESSION['user_db_id'] = $user['id'];
                // $_SESSION['authenticatedUserId'] = $user['username'];
                // $_SESSION['just_verified'] = true;
                
                $message = "✅ Your email has been verified! Welcome bonus of RM 50.00 added to your wallet. Redirecting to dashboard...";
                $redirect = false;
                
            } else {
                $message = "❌ Verification failed. Please try again.";
                SecurityUtils::logSecurityEvent($user['username'], 'email_verification', 'failure');
            }
        } else {
            $message = "❌ Invalid or expired verification link.";
            SecurityUtils::logSecurityEvent('unknown', 'invalid_verification_token', 'failure', ['token' => substr($token, 0, 8) . '...']);
        }
        
        $conn->close();
        
    } catch (Exception $e) {
        error_log("Verification error: " . $e->getMessage());
        $message = "❌ Verification failed due to system error. Error: " . $e->getMessage();
    }
} else {
    $message = "❌ No token provided.";
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Email Verification - Secure FacePay</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary: #3498db;
            --success: #38a169;
            --error: #e53e3e;
            --text-dark: #2d3748;
            --text-light: #718096;
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
            max-width: 500px;
        }
        
        .form-container {
            background-color: white;
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            text-align: center;
            animation: fadeUp 0.6s ease-out;
            position: relative;
            overflow: hidden;
        }
        
        .form-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary), #667eea);
        }
        
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .logo {
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
        
        .logo i {
            font-size: 32px;
            color: white;
        }
        
        h2 {
            font-size: 28px;
            margin-bottom: 30px;
            color: var(--text-dark);
            font-weight: 700;
        }
        
        .success-message {
            background-color: #f0fff4;
            color: var(--success);
            border: 1px solid #c6f6d5;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            text-align: left;
            font-weight: 500;
        }
        
        .error-message {
            background-color: #fff5f5;
            color: var(--error);
            border: 1px solid #fed7d7;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            text-align: left;
            font-weight: 500;
        }
        
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .redirect-info {
            margin-top: 20px;
            padding: 15px;
            background-color: #e3f2fd;
            border-radius: 10px;
            color: var(--primary);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            font-weight: 500;
        }
        
        .action-links {
            margin-top: 30px;
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .btn {
            background: linear-gradient(135deg, var(--primary), #667eea);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(52, 152, 219, 0.3);
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, var(--text-light), #4a5568);
        }
        
        .btn-secondary:hover {
            box-shadow: 0 8px 25px rgba(74, 85, 104, 0.3);
        }
        
        @media (max-width: 480px) {
            .form-container {
                padding: 30px 25px;
            }
            
            .action-links {
                flex-direction: column;
            }
        }
    </style>
    
    <?php if ($redirect): ?>
    <script>
        let countdown = 5;
        
        function updateCountdown() {
            const countdownElement = document.getElementById('countdown');
            if (countdownElement) {
                countdownElement.textContent = countdown;
            }
            
            if (countdown <= 0) {
                window.location.href = 'dashboard.php';
            } else {
                countdown--;
                setTimeout(updateCountdown, 1000);
            }
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            updateCountdown();
        });
    </script>
    <?php endif; ?>
</head>
<body>
    <div class="container">
        <div class="form-container">
            <div class="logo">
                <i class="fas fa-envelope-check"></i>
            </div>
            
            <h2>Email Verification</h2>
            
            <div class="<?= strpos($message, '✅') !== false ? 'success-message' : 'error-message' ?>">
                <i class="fas fa-<?= strpos($message, '✅') !== false ? 'check-circle' : 'exclamation-circle' ?>"></i>
                <span><?= $message ?></span>
            </div>
            
            <?php if ($redirect): ?>
                <div class="redirect-info">
                    <div class="loading-spinner"></div>
                    Redirecting to dashboard in <span id="countdown">5</span> seconds...
                </div>
            <?php endif; ?>
            
            <div class="action-links">
                <?php if ($redirect): ?>
                    <a href="dashboard.php" class="btn">
                        <i class="fas fa-tachometer-alt"></i> Go to Dashboard Now
                    </a>
                <?php else: ?>
                    <a href="login.php" class="btn">
                        <i class="fas fa-sign-in-alt"></i> Login
                    </a>
                    <a href="register.php" class="btn btn-secondary">
                        <i class="fas fa-user-plus"></i> Register Again
                    </a>
                <?php endif; ?>
            </div>
        </div>
    </div>
</body>
</html>