<?php
// profile.php - Profile Management with Notification Emails
require_once 'config.php';

// Start session and check authentication
if (!SessionManager::start()) {
    header("Location: login.php");
    exit();
}

// Require basic authentication
if (!SessionManager::requireAuthLevel('basic')) {
    header("Location: login.php");
    exit();
}

$user_id = $_SESSION['authenticatedUserId'];
$username = $_SESSION['username'];
$user_db_id = $_SESSION['user_db_id'] ?? null;

$success = "";
$error = "";
$tab = $_GET['tab'] ?? 'general';

try {
    $conn = dbConnect();
    
    // Get user details
    $stmt = $conn->prepare("SELECT * FROM users WHERE username = ? OR user_id = ?");
    $stmt->bind_param("ss", $username, $user_id);
    $stmt->execute();
    $result = $stmt->get_result();
    $user = $result->fetch_assoc();
    $stmt->close();
    
    if (!$user) {
        session_destroy();
        header("Location: login.php");
        exit();
    }
    
    $user_db_id = $user['id'];
    
    // Check if face is registered
    $face_registered = false;
    $face_data = null;
    
    $check_table = $conn->query("SHOW TABLES LIKE 'face_embeddings'");
    if ($check_table->num_rows > 0) {
        $stmt = $conn->prepare("SELECT * FROM face_embeddings WHERE user_id = ?");
        $stmt->bind_param("s", $user['user_id']);
        $stmt->execute();
        $result = $stmt->get_result();
        $face_data = $result->fetch_assoc();
        $stmt->close();
        
        $face_registered = $face_data ? true : false;
    }
    
    // Handle form submissions
    if ($_SERVER["REQUEST_METHOD"] == "POST") {
        // CSRF validation
        if (!validateCSRFToken($_POST['csrf_token'] ?? '')) {
            $error = "Invalid request. Please try again.";
        } else {
            $action = $_POST['action'] ?? '';
            
            switch ($action) {
                case 'update_profile':
                    $full_name = SecurityUtils::sanitizeInput($_POST['full_name'] ?? '');
                    
                    if (empty($full_name)) {
                        $error = "Full name is required.";
                    } elseif ($full_name === $user['full_name']) {
                        $error = "No changes detected.";
                    } else {
                        $stmt = $conn->prepare("UPDATE users SET full_name = ? WHERE id = ?");
                        $stmt->bind_param("si", $full_name, $user_db_id);
                        
                        if ($stmt->execute()) {
                            $success = "Profile updated successfully!";
                            $user['full_name'] = $full_name;
                            
                            // Send notification email
                            sendNotificationEmail($user['email'], $user['full_name'], 'profile_update', [
                                'old_name' => $user['full_name'],
                                'new_name' => $full_name
                            ]);
                            
                            // Log security event
                            SecurityUtils::logSecurityEvent($user['user_id'], 'profile_update', 'success');
                        } else {
                            $error = "Failed to update profile.";
                        }
                        $stmt->close();
                    }
                    break;
                    
                case 'change_password':
                    $current_password = PasswordDeobfuscator::getPassword('current_password');
                    $new_password = PasswordDeobfuscator::getPassword('new_password');
                    $confirm_password = PasswordDeobfuscator::getPassword('confirm_password');
                    
                    if (empty($current_password) || empty($new_password) || empty($confirm_password)) {
                        $error = "All password fields are required.";
                    } elseif ($new_password !== $confirm_password) {
                        $error = "New passwords do not match.";
                    } elseif (!preg_match('/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^a-zA-Z\d]).{8,}$/', $new_password)) {
                        $error = "New password must be at least 8 characters and include uppercase, lowercase, number, and symbol.";
                    } else {
                        if (!password_verify($current_password, $user['password'])) {
                            $error = "Current password is incorrect.";
                        } else {
                            $password_hash = password_hash($new_password, PASSWORD_DEFAULT);
                            $stmt = $conn->prepare("UPDATE users SET password = ? WHERE id = ?");
                            $stmt->bind_param("si", $password_hash, $user_db_id);
                            
                            if ($stmt->execute()) {
                                $success = "Password changed successfully!";
                                
                                // Send notification email
                                sendNotificationEmail($user['email'], $user['full_name'], 'password_change');
                                
                                SecurityUtils::logSecurityEvent($user['user_id'], 'password_change', 'success');
                            } else {
                                $error = "Failed to change password.";
                            }
                            $stmt->close();
                        }
                    }
                    break;
                    
                case 'change_pin':
                    $current_pin = PasswordDeobfuscator::getPassword('current_pin');
                    $new_pin_array = PasswordDeobfuscator::getPin();
                    $new_pin = implode('', $new_pin_array);
                    
                    if (empty($current_pin) || empty($new_pin)) {
                        $error = "Current PIN and new PIN are required.";
                    } elseif (!preg_match('/^\d{6}$/', $new_pin)) {
                        $error = "New PIN must be exactly 6 digits.";
                    } else {
                        if (!password_verify($current_pin, $user['pin_code'])) {
                            $error = "Current PIN is incorrect.";
                        } else {
                            $pin_hash = password_hash($new_pin, PASSWORD_DEFAULT);
                            $stmt = $conn->prepare("UPDATE users SET pin_code = ? WHERE id = ?");
                            $stmt->bind_param("si", $pin_hash, $user_db_id);
                            
                            if ($stmt->execute()) {
                                $success = "PIN changed successfully!";
                                
                                // Send notification email
                                sendNotificationEmail($user['email'], $user['full_name'], 'pin_change');
                                
                                SecurityUtils::logSecurityEvent($user['user_id'], 'pin_change', 'success');
                            } else {
                                $error = "Failed to change PIN.";
                            }
                            $stmt->close();
                        }
                    }
                    break;
                    
                case 'remove_face':
                    if ($face_registered) {
                        $stmt = $conn->prepare("DELETE FROM face_embeddings WHERE user_id = ?");
                        $stmt->bind_param("s", $user['user_id']);
                        
                        if ($stmt->execute()) {
                            $success = "Face registration removed successfully!";
                            $face_registered = false;
                            
                            // Send notification email
                            sendNotificationEmail($user['email'], $user['full_name'], 'face_removal');
                            
                            SecurityUtils::logSecurityEvent($user['user_id'], 'face_removal', 'success');
                        } else {
                            $error = "Failed to remove face registration.";
                        }
                        $stmt->close();
                    }
                    break;
            }
        }
    }
    
    $conn->close();
    
} catch (Exception $e) {
    error_log("Profile error: " . $e->getMessage());
    $error = "Unable to load profile data. Error: " . $e->getMessage();
}

$csrf_token = generateCSRFToken();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Profile Settings - Secure FacePay</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary: #3498db;
            --primary-light: #e3f2fd;
            --primary-dark: #2980b9;
            --secondary: #f8f9fa;
            --success: #38a169;
            --warning: #ed8936;
            --error: #e53e3e;
            --info: #3182ce;
            --text-dark: #2d3748;
            --text-medium: #4a5568;
            --text-light: #718096;
            --border: #e2e8f0;
            --bg-light: #f7fafc;
            --shadow: rgba(0,0,0,0.1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-light);
            color: var(--text-dark);
            line-height: 1.6;
        }
        
        /* Header */
        .header {
            background: linear-gradient(135deg, var(--primary), #667eea);
            color: white;
            padding: 0;
            box-shadow: 0 2px 10px var(--shadow);
        }
        
        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 70px;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 20px;
            font-weight: 700;
        }
        
        .logo i {
            font-size: 24px;
        }
        
        .breadcrumb {
            display: flex;
            align-items: center;
            gap: 10px;
            color: rgba(255,255,255,0.8);
        }
        
        .breadcrumb a {
            color: white;
            text-decoration: none;
            transition: opacity 0.3s;
        }
        
        .breadcrumb a:hover {
            opacity: 0.8;
        }
        
        /* Main Content */
        .main-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        
        .profile-header {
            background: white;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            display: flex;
            align-items: center;
            gap: 25px;
        }
        
        .profile-avatar {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, var(--primary), #667eea);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 32px;
            font-weight: 700;
            color: white;
        }
        
        .profile-info h1 {
            font-size: 28px;
            font-weight: 700;
            color: var(--text-dark);
            margin-bottom: 5px;
        }
        
        .profile-info p {
            color: var(--text-light);
            font-size: 16px;
        }
        
        .face-status {
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }
        
        .face-registered {
            background: rgba(56, 161, 105, 0.1);
            color: var(--success);
        }
        
        .face-not-registered {
            background: rgba(237, 137, 54, 0.1);
            color: var(--warning);
        }
        
        /* Tabs */
        .tabs {
            background: white;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            overflow: hidden;
        }
        
        .tab-navigation {
            display: flex;
            border-bottom: 1px solid var(--border);
        }
        
        .tab-button {
            flex: 1;
            padding: 20px;
            background: none;
            border: none;
            font-size: 16px;
            font-weight: 600;
            color: var(--text-medium);
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        .tab-button:hover {
            background: var(--bg-light);
        }
        
        .tab-button.active {
            color: var(--primary);
            background: var(--primary-light);
            border-bottom: 3px solid var(--primary);
        }
        
        .tab-content {
            padding: 30px;
        }
        
        .tab-panel {
            display: none;
        }
        
        .tab-panel.active {
            display: block;
        }
        
        /* Forms */
        .form-section {
            margin-bottom: 40px;
        }
        
        .form-section:last-child {
            margin-bottom: 0;
        }
        
        .section-title {
            font-size: 20px;
            font-weight: 600;
            color: var(--text-dark);
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .form-group {
            margin-bottom: 20px;
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
        
        .input-group input:disabled {
            background-color: #e2e8f0;
            color: var(--text-light);
            cursor: not-allowed;
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
        
        /* PIN Input */
        .pin-box {
            display: flex;
            gap: 10px;
            justify-content: flex-start;
        }
        
        .pin-box input {
            width: 45px;
            height: 45px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
            border: 2px solid var(--border);
            border-radius: 8px;
            background-color: #fafafa;
            transition: all 0.3s ease;
        }
        
        .pin-box input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
            outline: none;
            background-color: white;
        }
        
        /* Messages */
        .message {
            padding: 15px 18px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 500;
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
        
        /* Buttons */
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--primary), #667eea);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(52, 152, 219, 0.3);
        }
        
        .btn-secondary {
            background: var(--secondary);
            color: var(--text-medium);
            border: 2px solid var(--border);
        }
        
        .btn-secondary:hover {
            background: var(--border);
        }
        
        .btn-danger {
            background: linear-gradient(135deg, var(--error), #c0392b);
            color: white;
        }
        
        .btn-danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(229, 62, 62, 0.3);
        }
        
        .btn-success {
            background: linear-gradient(135deg, var(--success), #27ae60);
            color: white;
        }
        
        .btn-success:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(56, 161, 105, 0.3);
        }
        
        .btn-warning {
            background: linear-gradient(135deg, var(--warning), #f56500);
            color: white;
        }
        
        .btn-warning:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(237, 137, 54, 0.3);
        }
        
        /* Face Registration Card */
        .face-card {
            background: linear-gradient(135deg, #667eea, var(--primary));
            color: white;
            padding: 25px;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 20px;
        }
        
        .face-icon {
            width: 80px;
            height: 80px;
            background: rgba(255,255,255,0.2);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 32px;
            margin: 0 auto 20px;
        }
        
        .button-group {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        /* Security Notice */
        .security-notice {
            background: rgba(56, 161, 105, 0.1);
            border: 1px solid rgba(56, 161, 105, 0.3);
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
            font-size: 13px;
            color: var(--success);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .header-content {
                padding: 0 15px;
                flex-direction: column;
                height: auto;
                gap: 15px;
                padding-top: 15px;
                padding-bottom: 15px;
            }
            
            .main-content {
                padding: 20px 15px;
            }
            
            .profile-header {
                flex-direction: column;
                text-align: center;
                gap: 15px;
            }
            
            .face-status {
                margin-left: 0;
            }
            
            .tab-navigation {
                flex-wrap: wrap;
            }
            
            .tab-button {
                flex: none;
                min-width: 50%;
            }
            
            .form-grid {
                grid-template-columns: 1fr;
            }
            
            .button-group {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="header-content">
            <div class="logo">
                <i class="fas fa-shield-alt"></i>
                SecureFacePay
            </div>
            
            <div class="breadcrumb">
                <a href="dashboard.php">
                    <i class="fas fa-home"></i> Dashboard
                </a>
                <i class="fas fa-chevron-right"></i>
                <span>Profile Settings</span>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="main-content">
        <!-- Profile Header -->
        <div class="profile-header">
            <div class="profile-avatar">
                <?= strtoupper(substr($user['full_name'] ?? $username, 0, 1)) ?>
            </div>
            <div class="profile-info">
                <h1><?= htmlspecialchars($user['full_name'] ?? $username) ?></h1>
                <p><?= htmlspecialchars($user['email']) ?></p>
            </div>
            <div class="face-status <?= $face_registered ? 'face-registered' : 'face-not-registered' ?>">
                <i class="fas fa-<?= $face_registered ? 'check-circle' : 'exclamation-triangle' ?>"></i>
                <?= $face_registered ? 'Face Registered' : 'Face Not Registered' ?>
            </div>
        </div>

        <!-- Messages -->
        <?php if ($error): ?>
            <div class="message error-message">
                <i class="fas fa-exclamation-circle"></i>
                <?= htmlspecialchars($error) ?>
            </div>
        <?php endif; ?>
        
        <?php if ($success): ?>
            <div class="message success-message">
                <i class="fas fa-check-circle"></i>
                <?= htmlspecialchars($success) ?>
            </div>
        <?php endif; ?>

        <!-- Tabs -->
        <div class="tabs">
            <div class="tab-navigation">
                <button class="tab-button <?= $tab === 'general' ? 'active' : '' ?>" onclick="switchTab('general')">
                    <i class="fas fa-user"></i> General
                </button>
                <button class="tab-button <?= $tab === 'security' ? 'active' : '' ?>" onclick="switchTab('security')">
                    <i class="fas fa-lock"></i> Security
                </button>
                <button class="tab-button <?= $tab === 'face' ? 'active' : '' ?>" onclick="switchTab('face')">
                    <i class="fas fa-camera"></i> Face Recognition
                </button>
            </div>

            <div class="tab-content">
                <!-- General Tab -->
                <div id="general" class="tab-panel <?= $tab === 'general' ? 'active' : '' ?>">
                    <!-- Profile Update Section -->
                    <div class="form-section">
                        <h2 class="section-title">
                            <i class="fas fa-user-edit"></i>
                            Personal Information
                        </h2>
                        
                        <form method="POST" id="profileForm">
                            <input type="hidden" name="csrf_token" value="<?= $csrf_token ?>">
                            <input type="hidden" name="action" value="update_profile">
                            
                            <div class="form-group">
                                <label for="full_name">Full Name</label>
                                <div class="input-group">
                                    <input type="text" id="full_name" name="full_name" 
                                           value="<?= htmlspecialchars($user['full_name']) ?>" required>
                                    <i class="fas fa-user input-icon"></i>
                                </div>
                            </div>
                            
                            <div class="form-group">
                                <label for="username">Username</label>
                                <div class="input-group">
                                    <input type="text" id="username" value="<?= htmlspecialchars($user['username']) ?>" disabled>
                                    <i class="fas fa-at input-icon"></i>
                                </div>
                                <small style="color: var(--text-light); font-size: 12px;">Username cannot be changed</small>
                            </div>
                            
                            <div class="security-notice">
                                <i class="fas fa-info-circle"></i>
                                A notification email will be sent when you update your profile
                            </div>
                            
                            <br><button type="submit" class="btn btn-primary">
                                <i class="fas fa-save"></i> Save Changes
                            </button>
                        </form>
                    </div>

                    <!-- Email Section -->
                    <div class="form-section">
                        <h2 class="section-title">
                            <i class="fas fa-envelope"></i>
                            Email Address Management
                        </h2>
                        
                        <div class="form-group">
                            <label for="current_email">Current Email</label>
                            <div class="input-group">
                                <input type="email" id="current_email" value="<?= htmlspecialchars($user['email']) ?>" disabled>
                                <i class="fas fa-envelope input-icon"></i>
                            </div>
                            <small style="color: var(--text-light); font-size: 12px;">Email Address cannot be changed</small>
                        </div>

                        <div class="security-notice">
                            <i class="fas fa-shield-alt"></i>
                            Email Address cannot be changed for security reasons
                        </div>
                    </div>
                </div>

                <!-- Security Tab -->
                <div id="security" class="tab-panel <?= $tab === 'security' ? 'active' : '' ?>">
                    <!-- Change Password -->
                    <div class="form-section">
                        <h2 class="section-title">
                            <i class="fas fa-key"></i>
                            Change Password
                        </h2>
                        
                        <form method="POST" id="passwordForm">
                            <input type="hidden" name="csrf_token" value="<?= $csrf_token ?>">
                            <input type="hidden" name="action" value="change_password">
                            
                            <div class="form-group">
                                <label for="current_password">Current Password</label>
                                <div class="input-group">
                                    <input type="password" id="current_password" name="current_password" required>
                                    <i class="fas fa-lock input-icon"></i>
                                </div>
                            </div>
                            
                            <div class="form-grid">
                                <div class="form-group">
                                    <label for="new_password">New Password</label>
                                    <div class="input-group">
                                        <input type="password" id="new_password" name="new_password" required>
                                        <i class="fas fa-lock input-icon"></i>
                                    </div>
                                </div>
                                
                                <div class="form-group">
                                    <label for="confirm_password">Confirm New Password</label>
                                    <div class="input-group">
                                        <input type="password" id="confirm_password" name="confirm_password" required>
                                        <i class="fas fa-lock input-icon"></i>
                                    </div>
                                </div>
                            </div>

                            <div class="security-notice">
                                <i class="fas fa-info-circle"></i>
                                A notification email will be sent when your password is changed
                            </div>
                            
                            <br><button type="submit" class="btn btn-primary">
                                <i class="fas fa-key"></i> Change Password
                            </button>
                        </form>
                    </div>

                    <!-- Change PIN -->
                    <div class="form-section">
                        <h2 class="section-title">
                            <i class="fas fa-hashtag"></i>
                            Change Transaction PIN
                        </h2>
                        
                        <form method="POST" id="pinForm">
                            <input type="hidden" name="csrf_token" value="<?= $csrf_token ?>">
                            <input type="hidden" name="action" value="change_pin">
                            
                            <div class="form-group">
                                <label for="current_pin">Current PIN</label>
                                <div class="input-group">
                                    <input type="password" id="current_pin" name="current_pin" maxlength="6" pattern="\d{6}" required>
                                    <i class="fas fa-hashtag input-icon"></i>
                                </div>
                            </div>
                            
                            <div class="form-group">
                                <label>New PIN (6 digits)</label>
                                <div class="pin-box">
                                    <?php for ($i = 0; $i < 6; $i++): ?>
                                        <input type="password" name="new_pin[]" maxlength="1" pattern="\d" inputmode="numeric" required>
                                    <?php endfor; ?>
                                </div>
                            </div>

                            <div class="security-notice">
                                <i class="fas fa-info-circle"></i>
                                A notification email will be sent when your PIN is changed
                            </div>
                            
                            <br><button type="submit" class="btn btn-primary">
                                <i class="fas fa-hashtag"></i> Change PIN
                            </button>
                        </form>
                    </div>
                </div>

                <!-- Face Recognition Tab -->
                <div id="face" class="tab-panel <?= $tab === 'face' ? 'active' : '' ?>">
                    <div class="form-section">
                        <h2 class="section-title">
                            <i class="fas fa-camera"></i>
                            Facial Recognition Management
                        </h2>
                        
                        <div class="face-card">
                            <div class="face-icon">
                                <i class="fas fa-<?= $face_registered ? 'user-check' : 'user-slash' ?>"></i>
                            </div>
                            <h3><?= $face_registered ? 'Face Successfully Registered' : 'Face Not Registered' ?></h3>
                            <p>
                                <?= $face_registered ? 
                                    'Your face is registered for secure kiosk payments. You can re-register to update your facial data.' : 
                                    'Register your face to enable secure payments at kiosks and self-service machines.' ?>
                            </p>
                        </div>
                        
                        <div class="button-group">
                            <?php if (!$face_registered): ?>
                                <a href="setup_face.php" class="btn btn-success">
                                    <i class="fas fa-camera"></i> Register Face
                                </a>
                            <?php else: ?>
                                <a href="setup_face.php" class="btn btn-warning">
                                    <i class="fas fa-camera"></i> Re-register Face
                                </a>
                                
                                <form method="POST" style="display: inline-block;" onsubmit="return confirm('Are you sure you want to remove your face registration? You will need to re-register to use facial payments.')">
                                    <input type="hidden" name="csrf_token" value="<?= $csrf_token ?>">
                                    <input type="hidden" name="action" value="remove_face">
                                    <button type="submit" class="btn btn-danger">
                                        <i class="fas fa-trash"></i> Remove Face Registration
                                    </button>
                                </form>
                            <?php endif; ?>
                        </div>

                        <div class="security-notice">
                            <i class="fas fa-info-circle"></i>
                            A notification email will be sent for any face registration changes
                        </div>
                        
                        <?php if ($face_registered && $face_data): ?>
                            <div style="background: var(--bg-light); padding: 20px; border-radius: 12px; margin-top: 20px;">
                                <h4 style="margin-bottom: 10px; color: var(--text-dark);">
                                    <i class="fas fa-info-circle"></i> Registration Details
                                </h4>
                                <p style="color: var(--text-medium); font-size: 14px;">
                                    <strong>Registered:</strong> <?= isset($face_data['created_at']) ? date('M j, Y', strtotime($face_data['created_at'])) : 'Recently' ?><br>
                                    <strong>Status:</strong> <span style="color: var(--success);">Active and ready for kiosk payments</span>
                                </p>
                            </div>
                        <?php endif; ?>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Include centralized password security -->
    <script src="js/password-security.js"></script>

    <script>
        // Tab switching functionality
        function switchTab(tabName) {
            // Hide all tab panels
            document.querySelectorAll('.tab-panel').forEach(panel => {
                panel.classList.remove('active');
            });
            
            // Remove active class from all tab buttons
            document.querySelectorAll('.tab-button').forEach(button => {
                button.classList.remove('active');
            });
            
            // Show selected tab panel
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to selected tab button
            event.target.classList.add('active');
            
            // Update URL without refresh
            const url = new URL(window.location);
            url.searchParams.set('tab', tabName);
            window.history.pushState({}, '', url);
        }

        // PIN input handling
        const pinInputs = document.querySelectorAll("input[name='new_pin[]']");
        
        pinInputs.forEach((input, index) => {
            input.addEventListener("input", (e) => {
                // Only allow numbers
                e.target.value = e.target.value.replace(/[^0-9]/g, '');
                
                // Move to next input when current one is filled
                if (e.target.value.length === 1 && index < pinInputs.length - 1) {
                    pinInputs[index + 1].focus();
                }
            });
            
            input.addEventListener("keydown", (e) => {
                if (e.key === "Backspace") {
                    if (e.target.value === '' && index > 0) {
                        pinInputs[index - 1].focus();
                        pinInputs[index - 1].value = '';
                        e.preventDefault();
                    }
                }
                
                // Arrow key navigation
                if (e.key === "ArrowLeft" && index > 0) {
                    pinInputs[index - 1].focus();
                    e.preventDefault();
                }
                
                if (e.key === "ArrowRight" && index < pinInputs.length - 1) {
                    pinInputs[index + 1].focus();
                    e.preventDefault();
                }
            });
            
            // Handle paste
            input.addEventListener("paste", (e) => {
                e.preventDefault();
                const pasteData = e.clipboardData.getData('text').replace(/[^0-9]/g, '').slice(0, 6);
                
                for (let i = 0; i < Math.min(pasteData.length, pinInputs.length); i++) {
                    pinInputs[i].value = pasteData[i];
                }
                
                if (pasteData.length >= pinInputs.length) {
                    pinInputs[pinInputs.length - 1].focus();
                } else if (pasteData.length > 0) {
                    pinInputs[pasteData.length].focus();
                }
            });
        });

        // Password confirmation validation
        document.getElementById('passwordForm').addEventListener('submit', function(e) {
            const newPassword = document.getElementById('new_password').value;
            const confirmPassword = document.getElementById('confirm_password').value;
            
            if (newPassword !== confirmPassword) {
                e.preventDefault();
                alert('New passwords do not match!');
                return false;
            }
            
            if (!/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^a-zA-Z\d]).{8,}$/.test(newPassword)) {
                e.preventDefault();
                alert('Password must be at least 8 characters and include uppercase, lowercase, number, and symbol.');
                return false;
            }
        });

        // Form submission feedback
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', function() {
                const submitBtn = this.querySelector('button[type="submit"]');
                if (submitBtn && !submitBtn.disabled) {
                    const originalText = submitBtn.innerHTML;
                    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                    submitBtn.disabled = true;
                    
                    // Re-enable after 10 seconds to prevent permanent lockout
                    setTimeout(() => {
                        submitBtn.innerHTML = originalText;
                        submitBtn.disabled = false;
                    }, 10000);
                }
            });
        });

        // Auto-focus first input in active tab
        document.addEventListener('DOMContentLoaded', function() {
            const activeTab = document.querySelector('.tab-panel.active');
            if (activeTab) {
                const firstInput = activeTab.querySelector('input:not([disabled]):not([type="hidden"])');
                if (firstInput && window.innerWidth > 768) {
                    setTimeout(() => firstInput.focus(), 500);
                }
            }
        });

        // Clear messages after 7 seconds
        setTimeout(() => {
            const messages = document.querySelectorAll('.message');
            messages.forEach(message => {
                message.style.transition = 'opacity 0.5s ease';
                message.style.opacity = '0';
                setTimeout(() => message.remove(), 500);
            });
        }, 7000);
    </script>
</body>
</html>