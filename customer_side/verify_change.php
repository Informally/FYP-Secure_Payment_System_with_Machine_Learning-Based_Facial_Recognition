<?php
// verify_change.php - Handle email verification
require_once 'config.php';

if (!SessionManager::start()) {
    header("Location: login.php");
    exit();
}

$token = $_GET['token'] ?? '';
$success = false;
$error = '';

if (empty($token)) {
    $error = "Invalid verification link.";
} else {
    // Check password change
    if (isset($_SESSION['pending_password_change']) && 
        $_SESSION['pending_password_change']['token'] === $token &&
        $_SESSION['pending_password_change']['expires'] > time()) {
        
        $conn = dbConnect();
        $stmt = $conn->prepare("UPDATE users SET password = ? WHERE id = ?");
        $stmt->bind_param("si", $_SESSION['pending_password_change']['new_password'], $_SESSION['user_db_id']);
        
        if ($stmt->execute()) {
            $success = "Password changed successfully!";
            unset($_SESSION['pending_password_change']);
            SecurityUtils::logSecurityEvent($_SESSION['authenticatedUserId'], 'password_change', 'success');
        } else {
            $error = "Failed to update password.";
        }
        $stmt->close();
        $conn->close();
    }
    // Similar blocks for PIN and face changes...
    elseif (isset($_SESSION['pending_pin_change']) && 
            $_SESSION['pending_pin_change']['token'] === $token &&
            $_SESSION['pending_pin_change']['expires'] > time()) {
        
        $conn = dbConnect();
        $stmt = $conn->prepare("UPDATE users SET pin_code = ? WHERE id = ?");
        $stmt->bind_param("si", $_SESSION['pending_pin_change']['new_pin'], $_SESSION['user_db_id']);
        
        if ($stmt->execute()) {
            $success = "PIN changed successfully!";
            unset($_SESSION['pending_pin_change']);
            SecurityUtils::logSecurityEvent($_SESSION['authenticatedUserId'], 'pin_change', 'success');
        } else {
            $error = "Failed to update PIN.";
        }
        $stmt->close();
        $conn->close();
    }
    elseif (isset($_SESSION['pending_face_reregistration']) && 
            $_SESSION['pending_face_reregistration']['token'] === $token &&
            $_SESSION['pending_face_reregistration']['expires'] > time()) {
        
        $success = "Face re-registration verified! You can now register your face.";
        unset($_SESSION['pending_face_reregistration']);
        header("Location: setup_face.php?verified=1");
        exit();
    }
    else {
        $error = "Invalid or expired verification link.";
    }
}
?>

<!DOCTYPE html>
<html>
<head>
    <title>Email Verification - SecureFacePay</title>
    <style>/* Add your styles here */</style>
</head>
<body>
    <div class="container">
        <?php if ($success): ?>
            <div class="success"><?= htmlspecialchars($success) ?></div>
            <a href="profile.php">Return to Profile</a>
        <?php else: ?>
            <div class="error"><?= htmlspecialchars($error) ?></div>
            <a href="profile.php">Return to Profile</a>
        <?php endif; ?>
    </div>
</body>
</html>