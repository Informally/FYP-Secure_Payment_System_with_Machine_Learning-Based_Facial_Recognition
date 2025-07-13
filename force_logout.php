<?php
session_start();
session_unset();
session_destroy();

echo "Session cleared. <a href='index.php'>Go back</a>";
?>
