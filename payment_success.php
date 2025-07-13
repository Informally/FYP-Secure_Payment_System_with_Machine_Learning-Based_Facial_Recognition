<?php
session_start();
require_once 'config.php';

if (!isset($_SESSION['user_id'])) {
    header("Location: scan.php");
    exit();
}

$transaction_id = $_GET['transaction_id'] ?? '';
$conn = dbConnect();
$user_id = $_SESSION['user_id'];

$stmt = $conn->prepare("SELECT amount, merchant_name, timestamp FROM transactions WHERE user_id = ? AND transaction_id = ?");
$stmt->bind_param("is", $user_id, $transaction_id);
$stmt->execute();
$result = $stmt->get_result();
$transaction = $result->fetch_assoc();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Payment Successful</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link rel="stylesheet" href="style.css">
</head>
<body>
<div class="container">
    <div class="form-container">
        <h2><i class="fas fa-check-circle"></i> Payment Successful</h2>
        <div class="success-message">
            <p>Your payment was processed successfully!</p>
            <?php if ($transaction): ?>
                <p><strong>Transaction ID:</strong> <?php echo htmlspecialchars($transaction_id); ?></p>
                <p><strong>Amount:</strong> RM<?php echo number_format($transaction['amount'], 2); ?></p>
                <p><strong>Merchant:</strong> <?php echo htmlspecialchars($transaction['merchant_name']); ?></p>
                <p><strong>Date:</strong> <?php echo date("d M Y, H:i", strtotime($transaction['timestamp'])); ?></p>
            <?php else: ?>
                <p>Transaction details not found.</p>
            <?php endif; ?>
        </div>
        <div class="form-footer">
            <a href="dashboard.php">Back to Dashboard</a>
        </div>
    </div>
</div>
</body>
</html>