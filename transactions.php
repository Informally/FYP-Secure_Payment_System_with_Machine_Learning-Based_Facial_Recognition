<?php
session_start();
require_once 'config.php';

if (!isset($_SESSION['user_id'])) {
    header("Location: scan.php");
    exit();
}

$conn = dbConnect();
$user_id = $_SESSION['user_id'];

$stmt = $conn->prepare("SELECT amount, status, timestamp FROM transactions WHERE user_id = ? ORDER BY timestamp DESC");
$stmt->bind_param("i", $user_id);
$stmt->execute();
$result = $stmt->get_result();
$transactions = $result->fetch_all(MYSQLI_ASSOC);
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Transaction History</title>
    <link rel="stylesheet" href="style.css">
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 10px;
            border: 1px solid #ddd;
            text-align: center;
        }
        th {
            background-color: #3498db;
            color: white;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="form-container">
        <h2>Your Transactions</h2>
        <?php if (empty($transactions)): ?>
            <p>No transactions found.</p>
        <?php else: ?>
            <table>
                <thead>
                    <tr>
                        <th>Amount (RM)</th>
                        <th>Status</th>
                        <th>Date</th>
                    </tr>
                </thead>
                <tbody>
                    <?php foreach ($transactions as $tx): ?>
                        <tr>
                            <td><?= number_format($tx['amount'], 2) ?></td>
                            <td><?= htmlspecialchars($tx['status']) ?></td>
                            <td><?= date("d M Y, H:i", strtotime($tx['timestamp'])) ?></td>
                        </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
        <?php endif; ?>
        <div class="form-footer">
            <a href="dashboard.php">Back to Dashboard</a>
        </div>
    </div>
</div>
</body>
</html>
