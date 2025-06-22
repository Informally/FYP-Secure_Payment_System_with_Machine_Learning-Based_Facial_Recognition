<?php
require_once 'config/admin_config.php';

echo "Testing authentication...<br>";
echo "Session token: " . ($_SESSION['admin_session_token'] ?? 'NONE') . "<br>";

try {
    $admin = requireAdminAuth();
    echo "✅ Auth successful!<br>";
    echo "Admin: " . $admin['username'] . " (" . $admin['role'] . ")";
} catch (Exception $e) {
    echo "❌ Auth failed: " . $e->getMessage();
}
?>