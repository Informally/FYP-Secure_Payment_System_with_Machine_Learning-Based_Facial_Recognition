<?php
echo "JWT_SECRET: " . bin2hex(random_bytes(32)) . "\n";
echo "ENCRYPTION_KEY: " . bin2hex(random_bytes(16)) . "\n";
?>