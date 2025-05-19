<?php
// AES Key and IV (keep these safe!)
define('AES_KEY', base64_decode('0KvocVvCxtHUCXBQ8LVuztXHBFy7fiyjzhxOsXxQCmU='));
define('AES_IV', base64_decode('0OUi5xYbQkZgId+yqOrCUQ=='));

// MySQL Connection
define('DB_HOST', 'localhost');
define('DB_USER', 'root');
define('DB_PASS', '');
define('DB_NAME', 'payment_facial');

function dbConnect() {
    return new mysqli(DB_HOST, DB_USER, DB_PASS, DB_NAME);
}