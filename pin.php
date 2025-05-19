<?php
session_start();
require_once 'config.php';
require_once 'encrypt.php';

if (!isset($_SESSION['user_id'])) {
    header("Location: scan.php");
    exit();
}

$error = "";

if ($_SERVER["REQUEST_METHOD"] === "POST") {
    $conn = dbConnect();
    $user_id = $_SESSION['user_id'];
    $pin_input = implode('', $_POST['pin'] ?? []);

    $stmt = $conn->prepare("SELECT pin_code FROM users WHERE id = ?");
    $stmt->bind_param("i", $user_id);
    $stmt->execute();
    $stmt->bind_result($encrypted_pin);
    $stmt->fetch();
    $stmt->close();

    if (aes_decrypt($encrypted_pin) === $pin_input) {
        header("Location: payment.php");
        exit();
    } else {
        $error = "Invalid PIN. Please try again.";
    }
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Enter PIN</title>
    <link rel="stylesheet" href="style.css">
    <style>
        .pin-box {
            display: flex;
            justify-content: center;
            gap: 10px;
        }
        .pin-box input {
            width: 40px;
            font-size: 24px;
            text-align: center;
            -webkit-text-security: disc;
        }
    </style>
    <script>
        document.addEventListener("DOMContentLoaded", () => {
            const inputs = document.querySelectorAll(".pin-box input");
            inputs.forEach((input, index) => {
                input.addEventListener("input", () => {
                    if (input.value.length === 1 && index < inputs.length - 1) {
                        inputs[index + 1].focus();
                    }
                });
            });
        });
    </script>
</head>
<body>
<div class="container">
    <div class="form-container">
        <h2>Enter Your 6-digit PIN</h2>
        <?php if (!empty($error)): ?><div class="error-message"><?= htmlspecialchars($error) ?></div><?php endif; ?>
        <form method="POST">
            <div class="pin-box">
                <?php for ($i = 0; $i < 6; $i++): ?>
                    <input type="password" name="pin[]" maxlength="1" pattern="\d" required>
                <?php endfor; ?>
            </div>
            <br>
            <button type="submit">Confirm</button>
        </form>
    </div>
</div>
</body>
</html>
