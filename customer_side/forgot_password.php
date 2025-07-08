<?php
// forgot_password.php - Simple Password Reset System with Forced Re-login
require_once 'config.php';

// Check if user is already logged in
SessionManager::start();
if (isset($_SESSION['authenticatedUserId'])) {
    header("Location: dashboard.php");
    exit();
}

$error = "";
$success = "";
$step = isset($_GET['step']) ? $_GET['step'] : 'request'; // 'request' or 'reset'
$token = isset($_GET['token']) ? $_GET['token'] : '';

// Step 1: Request password reset (send email)
if ($_SERVER["REQUEST_METHOD"] == "POST" && $step == 'request') {
    // CSRF validation
    if (!validateCSRFToken($_POST['csrf_token'] ?? '')) {
        $error = "Invalid request. Please try again.";
    } else {
        $email = SecurityUtils::sanitizeInput($_POST['email'] ?? '');
        
        if (empty($email)) {
            $error = "Please enter your email address.";
        } elseif (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
            $error = "Please enter a valid email address.";
        } else {
            try {
                $conn = dbConnect();
                
                // Check if email exists and is verified
                $stmt = $conn->prepare("SELECT id, user_id, username, full_name, email, is_verified FROM users WHERE email = ?");
                $stmt->bind_param("s", $email);
                $stmt->execute();
                $result = $stmt->get_result();
                $user = $result->fetch_assoc();
                
                if ($user && $user['is_verified']) {
                    // Generate verification token (reuse existing verification_token field)
                    $verification_token = bin2hex(random_bytes(32));
                    
                    // Update user's verification token
                    $update_stmt = $conn->prepare("UPDATE users SET verification_token = ? WHERE id = ?");
                    $update_stmt->bind_param("si", $verification_token, $user['id']);
                    $update_stmt->execute();
                    
                    // Send password reset email using existing function
                    if (sendPasswordResetEmail($email, $user['full_name'], $verification_token)) {
                        $success = "Password reset instructions have been sent to your email address. Please check your inbox.";
                        SecurityUtils::logSecurityEvent($user['user_id'], 'forgot_password_email_sent', 'success');
                    } else {
                        $error = "Failed to send email. Please try again later.";
                    }
                } else {
                    // Don't reveal if email exists for security
                    $success = "If an account with that email exists, password reset instructions have been sent.";
                }
                
                $conn->close();
                
            } catch (Exception $e) {
                error_log("Forgot password error: " . $e->getMessage());
                $error = "An error occurred. Please try again later.";
            }
        }
    }
}

// Step 2: Reset password (from email link)
if ($_SERVER["REQUEST_METHOD"] == "POST" && $step == 'reset') {
    // CSRF validation
    if (!validateCSRFToken($_POST['csrf_token'] ?? '')) {
        $error = "Invalid request. Please try again.";
    } else {
        $new_password = PasswordDeobfuscator::getPassword('new_password');
        $confirm_password = PasswordDeobfuscator::getPassword('confirm_password');
        
        if (empty($new_password) || empty($confirm_password)) {
            $error = "Please fill in both password fields.";
        } elseif ($new_password !== $confirm_password) {
            $error = "Passwords do not match.";
        } elseif (!preg_match('/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^a-zA-Z\d]).{8,}$/', $new_password)) {
            $error = "Password must be at least 8 characters and include uppercase, lowercase, number, and symbol.";
        } else {
            try {
                $conn = dbConnect();
                
                // Verify token
                $stmt = $conn->prepare("SELECT id, user_id, username, full_name, email FROM users WHERE verification_token = ?");
                $stmt->bind_param("s", $token);
                $stmt->execute();
                $result = $stmt->get_result();
                $user = $result->fetch_assoc();
                
                if ($user) {
                    // Update password and clear token
                    $password_hash = password_hash($new_password, PASSWORD_DEFAULT);
                    $update_stmt = $conn->prepare("UPDATE users SET password = ?, verification_token = NULL WHERE id = ?");
                    $update_stmt->bind_param("si", $password_hash, $user['id']);
                    $update_stmt->execute();
                    
                    // 🔐 SECURITY: Invalidate any existing sessions for this user
                    // Clear any remember tokens for this user
                    $clear_remember = $conn->prepare("DELETE FROM remember_tokens WHERE user_id = ?");
                    $clear_remember->bind_param("i", $user['id']);
                    $clear_remember->execute();
                    
                    SecurityUtils::logSecurityEvent($user['user_id'], 'password_reset_success', 'success', [
                        'method' => 'forgot_password',
                        'forced_logout' => true
                    ]);
                    
                    $success = "Your password has been successfully updated! For security reasons, please login with your new password.";
                    $step = 'complete';
                } else {
                    $error = "Invalid or expired reset link.";
                }
                
                $conn->close();
                
            } catch (Exception $e) {
                error_log("Password reset error: " . $e->getMessage());
                $error = "An error occurred. Please try again later.";
            }
        }
    }
}

// Verify token for reset step
if ($step == 'reset' && !empty($token)) {
    try {
        $conn = dbConnect();
        $stmt = $conn->prepare("SELECT full_name, email FROM users WHERE verification_token = ?");
        $stmt->bind_param("s", $token);
        $stmt->execute();
        $result = $stmt->get_result();
        $user_data = $result->fetch_assoc();
        $conn->close();
        
        if (!$user_data) {
            $error = "Invalid or expired reset link.";
            $step = 'request';
        }
    } catch (Exception $e) {
        $error = "An error occurred. Please try again later.";
        $step = 'request';
    }
}

$csrf_token = generateCSRFToken();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title><?= $step == 'reset' ? 'Reset Password' : 'Forgot Password' ?> - Secure FacePay</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary: #3498db;
            --primary-light: #e3f2fd;
            --primary-dark: #2980b9;
            --success: #38a169;
            --error: #e53e3e;
            --warning: #ed8936;
            --text-dark: #2d3748;
            --text-medium: #4a5568;
            --text-light: #718096;
            --border: #e2e8f0;
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
            line-height: 1.5;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            width: 100%;
            max-width: 420px;
        }
        
        .form-container {
            background-color: white;
            border-radius: 16px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            padding: 40px 35px;
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
        
        .logo-section {
            text-align: center;
            margin-bottom: 35px;
        }
        
        .logo {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, var(--primary), #667eea);
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
            color: var(--text-dark);
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        
        .subtitle {
            color: var(--text-light);
            font-size: 14px;
            line-height: 1.4;
        }
        
        .form-group {
            margin-bottom: 24px;
        }
        
        label {
            display: block;
            font-size: 14px;
            font-weight: 600;
            color: var(--text-medium);
            margin-bottom: 8px;
        }
        
        .input-group {
            position: relative;
        }
        
        .input-group input {
            width: 100%;
            height: 48px;
            padding: 0 20px 0 50px;
            border: 2px solid var(--border);
            border-radius: 12px;
            font-size: 15px;
            transition: all 0.3s ease;
            color: var(--text-dark);
            background-color: #fafafa;
        }
        
        .input-group input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(52, 152, 219, 0.1);
            outline: none;
            background-color: white;
        }
        
        .input-icon {
            position: absolute;
            left: 18px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-light);
            font-size: 16px;
            pointer-events: none;
            transition: color 0.3s ease;
        }
        
        .input-group input:focus + .input-icon {
            color: var(--primary);
        }
        
        .btn {
            width: 100%;
            height: 50px;
            background: linear-gradient(135deg, var(--primary), #667eea);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            margin-bottom: 25px;
            text-decoration: none;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(52, 152, 219, 0.3);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-success {
            background: linear-gradient(135deg, var(--success), #48bb78);
        }
        
        .btn-success:hover {
            box-shadow: 0 8px 25px rgba(56, 161, 105, 0.3);
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, var(--text-light), #4a5568);
            margin-bottom: 15px;
        }
        
        .btn-secondary:hover {
            box-shadow: 0 8px 25px rgba(74, 85, 104, 0.3);
        }
        
        .message {
            padding: 15px 18px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 14px;
            display: flex;
            align-items: flex-start;
            gap: 10px;
            font-weight: 500;
            line-height: 1.4;
        }
        
        .error-message {
            background-color: #fff5f5;
            color: var(--error);
            border: 1px solid #fed7d7;
        }
        
        .success-message {
            background-color: #f0fff4;
            color: var(--success);
            border: 1px solid #c6f6d5;
        }
        
        .security-notice {
            background-color: #fff7ed;
            color: var(--warning);
            border: 1px solid #fed7aa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .form-footer {
            text-align: center;
            color: var(--text-light);
            font-size: 14px;
        }
        
        .form-footer a {
            color: var(--primary);
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s ease;
        }
        
        .form-footer a:hover {
            color: var(--primary-dark);
            text-decoration: underline;
        }
        
        .user-info {
            background-color: #f7fafc;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 25px;
            border-left: 4px solid var(--primary);
        }
        
        @media (max-width: 480px) {
            .form-container {
                padding: 30px 25px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="form-container">
            <!-- Step 1: Email Request -->
            <?php if ($step == 'request'): ?>
                <div class="logo-section">
                    <div class="logo">
                        <i class="fas fa-key"></i>
                    </div>
                    <h2>Forgot Password?</h2>
                    <p class="subtitle">Enter your email address and we'll send you instructions to reset your password.</p>
                </div>
                
                <?php if ($error): ?>
                    <div class="message error-message">
                        <i class="fas fa-exclamation-circle"></i>
                        <span><?= htmlspecialchars($error) ?></span>
                    </div>
                <?php endif; ?>
                
                <?php if ($success): ?>
                    <div class="message success-message">
                        <i class="fas fa-check-circle"></i>
                        <span><?= htmlspecialchars($success) ?></span>
                    </div>
                <?php else: ?>
                    <form method="POST">
                        <input type="hidden" name="csrf_token" value="<?= $csrf_token ?>">
                        
                        <div class="form-group">
                            <label for="email">Email Address</label>
                            <div class="input-group">
                                <input type="email" id="email" name="email" 
                                       placeholder="Enter your email address" required>
                                <i class="fas fa-envelope input-icon"></i>
                            </div>
                        </div>
                        
                        <button type="submit" class="btn">
                            <i class="fas fa-paper-plane"></i>
                            Send Reset Request
                        </button>
                    </form>
                <?php endif; ?>
            
            <!-- Step 2: Password Reset -->
            <?php elseif ($step == 'reset'): ?>
                <div class="logo-section">
                    <div class="logo">
                        <i class="fas fa-lock"></i>
                    </div>
                    <h2>Reset Password</h2>
                    <p class="subtitle">Create a new secure password for your account.</p>
                </div>
                
                <?php if (isset($user_data)): ?>
                    <div class="user-info">
                        <div style="font-weight: 600; margin-bottom: 4px;"><?= htmlspecialchars($user_data['full_name']) ?></div>
                        <div style="font-size: 14px; color: var(--text-medium);"><?= htmlspecialchars($user_data['email']) ?></div>
                    </div>
                <?php endif; ?>
                
                <?php if ($error): ?>
                    <div class="message error-message">
                        <i class="fas fa-exclamation-circle"></i>
                        <span><?= htmlspecialchars($error) ?></span>
                    </div>
                <?php endif; ?>
                
                <form method="POST">
                    <input type="hidden" name="csrf_token" value="<?= $csrf_token ?>">
                    
                    <div class="form-group">
                        <label for="new_password">New Password</label>
                        <div class="input-group">
                            <input type="password" id="new_password" name="new_password" 
                                   placeholder="Enter your new password" required>
                            <i class="fas fa-lock input-icon"></i>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="confirm_password">Confirm New Password</label>
                        <div class="input-group">
                            <input type="password" id="confirm_password" name="confirm_password" 
                                   placeholder="Confirm your new password" required>
                            <i class="fas fa-lock input-icon"></i>
                        </div>
                    </div>
                    
                    <button type="submit" class="btn">
                        <i class="fas fa-save"></i>
                        Update Password
                    </button>
                </form>
            
            <!-- Step 3: Complete -->
            <?php elseif ($step == 'complete'): ?>
                <div class="logo-section">
                    <div class="logo" style="background: linear-gradient(135deg, var(--success), #48bb78);">
                        <i class="fas fa-check"></i>
                    </div>
                    <h2>Password Updated!</h2>
                    <p class="subtitle">Your password has been successfully reset.</p>
                </div>
                
                <div class="message success-message">
                    <i class="fas fa-check-circle"></i>
                    <span><?= htmlspecialchars($success) ?></span>
                </div>
                
                <div class="security-notice">
                    <i class="fas fa-shield-alt"></i>
                    <span>For security, all existing sessions have been invalidated. Please login with your new password.</span>
                </div>
                
                <a href="login.php" class="btn btn-success">
                    <i class="fas fa-sign-in-alt"></i>
                    Login with New Password
                </a>
                
                <a href="scan.php" class="btn btn-secondary">
                    <i class="fas fa-camera"></i>
                    Login with Face Recognition
                </a>
            <?php endif; ?>
            
            <div class="form-footer">
                <?php if ($step != 'complete'): ?>
                    <p>Remember your password? <a href="login.php">Sign In</a></p>
                    <p style="margin-top: 8px;">Or try <a href="scan.php">Face Recognition</a></p>
                <?php else: ?>
                    <p>Need help? <a href="mailto:support@facepay.com">Contact Support</a></p>
                <?php endif; ?>
            </div>
        </div>
    </div>
    
    <!-- Include centralized password security -->
    <script src="js/password-security.js"></script>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Focus first input
            const firstInput = document.querySelector('input[type="email"], input[type="password"]');
            if (firstInput) {
                firstInput.focus();
            }
            
            // Password confirmation validation
            const newPassword = document.getElementById('new_password');
            const confirmPassword = document.getElementById('confirm_password');
            
            if (newPassword && confirmPassword) {
                confirmPassword.addEventListener('input', function() {
                    if (this.value && newPassword.value !== this.value) {
                        this.style.borderColor = 'var(--error)';
                    } else {
                        this.style.borderColor = 'var(--border)';
                    }
                });
            }
        });
    </script>
</body>
</html>