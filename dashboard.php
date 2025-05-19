<?php
session_start();
require_once 'config.php';

if (!isset($_SESSION['user_id'])) {
    header("Location: scan.php");
    exit();
}

$conn = dbConnect();
$user_id = $_SESSION['user_id'];

// Fetch username
$stmt = $conn->prepare("SELECT username FROM users WHERE id = ?");
$stmt->bind_param("i", $user_id);
$stmt->execute();
$result = $stmt->get_result();
$user = $result->fetch_assoc();

// Fetch wallet balance
$stmt = $conn->prepare("SELECT balance FROM wallets WHERE user_id = ?");
$stmt->bind_param("i", $user_id);
$stmt->execute();
$result = $stmt->get_result();
$wallet = $result->fetch_assoc();

$balance = ($wallet && isset($wallet['balance'])) ? number_format($wallet['balance'], 2) : '0.00';
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Dashboard</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
<div class="container">
    <div class="form-container">
        <div class="dashboard-header">
            <div class="user-info">
                Welcome, <span><?= htmlspecialchars($user['username']) ?></span><br>
                Balance: <strong>RM<?= $balance ?></strong>
            </div>
            <form method="POST" action="logout.php">
                <button type="submit" class="logout-btn">Logout</button>
            </form>
        </div>

        <div class="card">
            <h3 class="card-title">Actions</h3>
            <div class="card-body">
                <p><a href="scan.php">Make a Payment</a></p>
                <p><a href="topup.php">Top-Up Wallet</a></p>
                <p><a href="transactions.php">View Transactions</a></p>
            </div>
        </div>
    </div>
</div>
</body>
</html>
