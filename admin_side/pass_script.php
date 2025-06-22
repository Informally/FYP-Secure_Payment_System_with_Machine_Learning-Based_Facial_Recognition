<?php
// update_admin_password.php
require_once '../customer_side/config.php';

$conn = dbConnect();

$username = 'admin';
$password = 'Admin123!';
$password_hash = password_hash($password, PASSWORD_DEFAULT);

// Just UPDATE the existing admin password instead of deleting
$stmt = $conn->prepare("UPDATE admin_users SET password_hash = ?, is_active = 1 WHERE username = ?");
$stmt->bind_param("ss", $password_hash, $username);

if ($stmt->execute()) {
    echo "✅ Admin password updated successfully!\n";
    echo "Username: admin\n";
    echo "Password: Admin123!\n";
} else {
    echo "❌ Error updating password: " . $stmt->error . "\n";
}

$stmt->close();
$conn->close();
?>