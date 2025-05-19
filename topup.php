<?php
session_start();
require_once 'config.php';

if (!isset($_SESSION['user_id'])) {
    header("Location: scan.php");
    exit();
}

$conn = dbConnect();
$user_id = $_SESSION['user_id'];
$message = "";

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $amount = floatval($_POST['amount']);
    if ($amount <= 0) {
        $message = "Top-up amount must be greater than 0.";
    } else {
        // Add amount to wallet
        $stmt = $conn->prepare("UPDATE wallets SET balance = balance + ? WHERE user_id = ?");
        $stmt->bind_param("di", $amount, $user_id);
        if ($stmt->execute()) {
            $message = "Successfully topped up RM$amount.";
        } else {
            $message = "Failed to update wallet.";
        }
    }
}

// Get current balance
$stmt = $conn->prepare("SELECT balance FROM wallets WHERE user_id = ?");
$stmt->bind_param("i", $user_id);
$stmt->execute();
$result = $stmt->get_result();
$balance = $result->fetch_assoc()['balance'];
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Top Up Wallet</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
<div class="container">
    <div class="form-container">
        <h2>Wallet Top-Up</h2>
        <p><strong>Current Balance:</strong> RM<?= number_format($balance, 2) ?></p>
        <?php if ($message): ?><div class="success-message"><?= htmlspecialchars($message) ?></div><?php endif; ?>
        <form method="POST">
            <div class="form-group">
                <label for="amount">Top-Up Amount (RM)</label>
                <input type="text" id="amount" name="amount" required pattern="\d+(\.\d{1,2})?">
            </div>
            <button type="submit">Top Up</button>
        </form>
        <div class="form-footer">
            <a href="dashboard.php">Back to Dashboard</a>
        </div>
    </div>
</div>
</body>
</html>