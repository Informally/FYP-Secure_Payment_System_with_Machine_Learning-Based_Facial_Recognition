<?php
require_once 'config.php';
session_start();

$token = $_GET['token'] ?? '';
$message = '';

if (!empty($token)) {
    $conn = dbConnect();

    $stmt = $conn->prepare("SELECT id FROM users WHERE verification_token = ?");
    $stmt->bind_param("s", $token);
    $stmt->execute();
    $stmt->store_result();

    if ($stmt->num_rows === 1) {
        $stmt->bind_result($user_id);
        $stmt->fetch();

        // Mark user as verified
        $update = $conn->prepare("UPDATE users SET is_verified = 1, verification_token = NULL WHERE id = ?");
        $update->bind_param("i", $user_id);
        if ($update->execute()) {
            $_SESSION['user_id'] = $user_id;
            $message = "✅ Your email has been verified! Redirecting to face registration...";
            $redirect = true;
        } else {
            $message = "❌ Verification failed. Please try again.";
        }
    } else {
        $message = "❌ Invalid or expired verification link.";
    }
} else {
    $message = "❌ No token provided.";
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Email Verification</title>
    <link rel="stylesheet" href="style.css">
    <?php if (!empty($redirect)): ?>
    <script>
        setTimeout(() => {
            window.location.href = 'register_face.php';
        }, 3000);
    </script>
    <?php endif; ?>
</head>
<body>
<div class="container">
    <div class="form-container">
        <h2>Email Verification</h2>
        <div class="<?= strpos($message, '✅') !== false ? 'success-message' : 'error-message' ?>">
            <?= $message ?>
        </div>
    </div>
</div>
</body>
</html>