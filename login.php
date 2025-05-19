<?php
require_once 'config.php';
require_once 'encrypt.php';
session_start();
$error = "";
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $conn = dbConnect();
    $username = trim($_POST['username']);
    $password = $_POST['password'];
    if (empty($username) || empty($password)) {
        $error = "Please fill in all fields.";
    } else {
        $stmt = $conn->prepare("SELECT id, password, is_verified FROM users WHERE username = ?");
        $stmt->bind_param("s", $username);
        $stmt->execute();
        $result = $stmt->get_result();
        if ($row = $result->fetch_assoc()) {
            if (!$row['is_verified']) {
                $error = "Please verify your email before logging in.";
            } else {
                $decrypted_password = aes_decrypt($row['password']);
                if ($decrypted_password === $password) {
                    $_SESSION['user_id'] = $row['id'];
                    header("Location: dashboard.php");
                    exit();
                } else {
                    $error = "Invalid username or password.";
                }
            }
        } else {
            $error = "Invalid username or password.";
        }
    }
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Facial Pay System</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link rel="stylesheet" href="style.css">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f0f4f8;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        
        .container {
            width: 100%;
            max-width: 450px;
            padding: 20px;
        }
        
        .form-container {
            background-color: #ffffff;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            padding: 40px;
            transition: all 0.3s ease;
            animation: fadeIn 0.6s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        h2 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 28px;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }
        
        h2 i {
            font-size: 28px;
            color: #3498db;
        }
        
        .form-group {
            margin-bottom: 24px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #4a5568;
            font-size: 15px;
        }
        
        input[type="text"],
        input[type="password"] {
            width: 100%;
            padding: 12px 15px;
            padding-left: 40px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            font-size: 16px;
            transition: all 0.3s;
            box-sizing: border-box;
        }
        
        input[type="text"]:focus,
        input[type="password"]:focus {
            border-color: #3498db;
            outline: none;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
        }
        
        button[type="submit"] {
            width: 100%;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 14px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 10px;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        button[type="submit"]:hover {
            background-color: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 7px 14px rgba(0, 0, 0, 0.1);
        }
        
        .form-footer {
            text-align: center;
            margin-top: 25px;
            color: #718096;
            font-size: 15px;
        }
        
        .form-footer a {
            color: #3498db;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s;
        }
        
        .form-footer a:hover {
            color: #2980b9;
            text-decoration: underline;
        }
        
        .error-message {
            background-color: #fee2e2;
            color: #dc2626;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 15px;
            display: flex;
            align-items: center;
        }
        
        .error-message:before {
            content: "\f071";
            font-family: "Font Awesome 6 Free";
            font-weight: 900;
            margin-right: 10px;
        }
        
        /* Input group with icons */
        .input-group {
            position: relative;
            margin-bottom: 20px;
        }
        
        .input-icon {
            position: absolute;
            left: 14px;
            top: 50%;
            transform: translateY(-50%);
            color: #a0aec0;
        }
        
        /* Alternative login options */
        .alternative-login {
            text-align: center;
            margin-top: 25px;
            position: relative;
        }
        
        .divider {
            display: flex;
            align-items: center;
            margin: 20px 0;
        }
        
        .divider:before,
        .divider:after {
            content: "";
            flex: 1;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .divider span {
            padding: 0 10px;
            color: #a0aec0;
            font-size: 14px;
        }
        
        .alt-login-btn {
            display: block;
            width: 100%;
            background-color: #f8fafc;
            color: #64748b;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            margin: 10px 0;
            transition: all 0.3s;
            text-align: center;
            text-decoration: none;
        }
        
        .alt-login-btn:hover {
            background-color: #f1f5f9;
            border-color: #cbd5e1;
        }
        
        .alt-login-btn i {
            margin-right: 8px;
        }
        
        /* Remember me & Forgot password */
        .login-options {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .remember-me {
            display: flex;
            align-items: center;
        }
        
        .remember-me input {
            margin-right: 5px;
        }
        
        .forgot-password {
            color: #3498db;
            text-decoration: none;
            font-size: 14px;
            transition: color 0.3s;
        }
        
        .forgot-password:hover {
            color: #2980b9;
            text-decoration: underline;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="form-container">
        <h2><i class="fas fa-user-shield"></i> Login</h2>
        
        <?php if ($error): ?>
            <div class="error-message"><?= htmlspecialchars($error) ?></div>
        <?php endif; ?>

        <form method="POST">
            <div class="form-group">
                <label for="username">Username</label>
                <div class="input-group">
                    <i class="fas fa-user input-icon"></i>
                    <input type="text" id="username" name="username" placeholder="Enter your username" required>
                </div>
            </div>
            
            <div class="form-group">
                <label for="password">Password</label>
                <div class="input-group">
                    <i class="fas fa-lock input-icon"></i>
                    <input type="password" id="password" name="password" placeholder="Enter your password" required>
                </div>
            </div>
            
            <div class="login-options">
                <div class="remember-me">
                    <input type="checkbox" id="remember" name="remember">
                    <label for="remember" style="display: inline; margin-bottom: 0;">Remember me</label>
                </div>
                <a href="forgot-password.php" class="forgot-password">Forgot password?</a>
            </div>
            
            <button type="submit"><i class="fas fa-sign-in-alt"></i> Login</button>
        </form>
        
        <div class="divider">
            <span>OR</span>
        </div>
        
        <a href="scan.php" class="alt-login-btn">
            <i class="fas fa-camera"></i> Login with Facial Recognition
        </a>
        
        <div class="form-footer">
            <p>Don't have an account? <a href="register.php">Register</a></p>
        </div>
    </div>
</div>

<script>
    // Focus the username field on page load
    document.addEventListener('DOMContentLoaded', function() {
        document.getElementById('username').focus();
    });
</script>
</body>
</html>