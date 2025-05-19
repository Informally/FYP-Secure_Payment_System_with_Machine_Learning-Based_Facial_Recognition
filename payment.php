<?php
session_start();
require_once 'config.php';

if (!isset($_SESSION['user_id'])) {
    header("Location: scan.php");
    exit();
}

$conn = dbConnect();
$user_id = $_SESSION['user_id'];

// Default values
$message = "";
$success = false;

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $amount = floatval($_POST['amount']);

    // Fetch user's wallet balance
    $stmt = $conn->prepare("SELECT balance FROM wallets WHERE user_id = ?");
    $stmt->bind_param("i", $user_id);
    $stmt->execute();
    $result = $stmt->get_result();

    if ($row = $result->fetch_assoc()) {
        $balance = floatval($row['balance']);

        if ($balance >= $amount) {
            // Deduct amount
            $stmt = $conn->prepare("UPDATE wallets SET balance = balance - ? WHERE user_id = ?");
            $stmt->bind_param("di", $amount, $user_id);
            $stmt->execute();

            // Log transaction
            $stmt = $conn->prepare("INSERT INTO transactions (user_id, amount, status) VALUES (?, ?, 'success')");
            $stmt->bind_param("id", $user_id, $amount);
            $stmt->execute();

            $message = "Payment of RM$amount was successful.";
            $success = true;
        } else {
            // Insufficient funds
            $message = "Insufficient funds. Your current balance is RM$balance.";
        }
    } else {
        $message = "Wallet not found for user.";
    }
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Payment</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
<div class="container">
    <div class="form-container">
        <h2>Payment Page</h2>
        <?php if ($message): ?>
            <div class="<?= $success ? 'success-message' : 'error-message' ?>">
                <?= htmlspecialchars($message) ?>
            </div>
            <?php if (!$success): ?>
                <form method="POST">
                    <input type="hidden" name="amount" value="<?= htmlspecialchars($_POST['amount'] ?? '') ?>">
                    <button type="submit">Retry Payment</button>
                </form>
                <div class="form-footer">
                    <a href="scan.php">Cancel and Go Back</a>
                </div>
            <?php endif; ?>
        <?php else: ?>
            <form method="POST">
                <div class="form-group">
                    <label for="amount">Enter Payment Amount (RM)</label>
                    <input type="text" id="amount" name="amount" required pattern="\d+(\.\d{1,2})?">
                </div>
                <button type="submit">Proceed Payment</button>
            </form>
        <?php endif; ?>
    </div>
</div>
</body>
</html>
