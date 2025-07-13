<?php
session_start();
require_once 'config.php';

if (!isset($_SESSION['user_id'])) {
    header("Location: scan.php");
    exit();
}

$error = $_GET['error'] ?? 'Unknown error occurred.';
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Payment Failed</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link rel="stylesheet" href="style.css">
</head>
<body>
<div class="container">
    <div class="form-container">
        <h2><i class="fas fa-exclamation-circle"></i> Payment Failed</h2>
        <div class="error-message">
            <p><?php echo htmlspecialchars($error); ?></p>
        </div>
        <div class="form-footer">
            <a href="business_integration/facialpay_checkout.php">Retry Payment</a> | <a href="business-integration/e-commerce.php">Cancel and Return to Store</a>
        </div>
    </div>
</div>
</body>
</html>