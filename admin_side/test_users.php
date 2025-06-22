<?php
// Create this as: admin_side/test_users.php
session_start();

echo "<h2>🧪 Simple Session Test</h2>";
echo "Session ID: " . session_id() . "<br>";
echo "Session Token: " . ($_SESSION['admin_session_token'] ?? 'NOT SET') . "<br>";

if (isset($_SESSION['admin_session_token'])) {
    echo "<h3>✅ Session exists! Let's test config...</h3>";
    
    try {
        require_once 'config/admin_config.php';
        echo "✅ Config loaded<br>";

        // DEBUG: Check what's in the database
        echo "<h3>🔍 Database Debug:</h3>";
        try {
            $conn = dbConnect();
            
            // Check if admin_sessions table exists and has data
            $result = $conn->query("SELECT COUNT(*) as count FROM admin_sessions");
            $count = $result->fetch_assoc()['count'];
            echo "Total sessions in database: " . $count . "<br>";
            
            // Check for your specific token
            $token = $_SESSION['admin_session_token'];
            echo "Looking for token: " . substr($token, 0, 20) . "...<br>";
            
            $stmt = $conn->prepare("SELECT admin_id, expires_at FROM admin_sessions WHERE session_token = ?");
            $stmt->bind_param("s", $token);
            $stmt->execute();
            $result = $stmt->get_result();
            
            if ($result->num_rows > 0) {
                $session_data = $result->fetch_assoc();
                echo "✅ Token found in database!<br>";
                echo "Admin ID: " . $session_data['admin_id'] . "<br>";
                echo "Expires at: " . $session_data['expires_at'] . "<br>";
                echo "Current time: " . date('Y-m-d H:i:s') . "<br>";
                
                // Check if expired
                if ($session_data['expires_at'] > date('Y-m-d H:i:s')) {
                    echo "✅ Session not expired<br>";
                    // DEBUG: Check admin_users table
                    echo "<h3>🔍 Admin User Debug:</h3>";
                    $admin_id = $session_data['admin_id'];
                    echo "Looking for admin ID: " . $admin_id . "<br>";

                    $stmt = $conn->prepare("SELECT id, username, full_name, role, is_active FROM admin_users WHERE id = ?");
                    $stmt->bind_param("i", $admin_id);
                    $stmt->execute();
                    $admin_result = $stmt->get_result();

                    if ($admin_result->num_rows > 0) {
                        $admin_data = $admin_result->fetch_assoc();
                        echo "✅ Admin user found!<br>";
                        echo "Username: " . $admin_data['username'] . "<br>";
                        echo "Full Name: " . $admin_data['full_name'] . "<br>";
                        echo "Role: " . $admin_data['role'] . "<br>";
                        echo "Is Active: " . ($admin_data['is_active'] ? 'YES' : 'NO') . "<br>";
                        echo "Is Active: YES<br>";

                        // DEBUG: Test the exact validateSession query
                        echo "<h3>🔍 ValidateSession Query Debug:</h3>";
                        $token = $_SESSION['admin_session_token'];

                        $debug_query = "
                            SELECT s.admin_id, s.expires_at, a.username, a.email, a.role, a.full_name, a.is_active
                            FROM admin_sessions s
                            JOIN admin_users a ON s.admin_id = a.id
                            WHERE s.session_token = ? AND s.expires_at > NOW() AND a.is_active = TRUE
                        ";

                        echo "Query: " . str_replace('?', "'" . substr($token, 0, 20) . "...'", $debug_query) . "<br>";

                        $stmt = $conn->prepare($debug_query);
                        $stmt->bind_param("s", $token);
                        $stmt->execute();
                        $validation_result = $stmt->get_result();

                        echo "Number of rows returned: " . $validation_result->num_rows . "<br>";

                        if ($validation_result->num_rows > 0) {
                            $validation_data = $validation_result->fetch_assoc();
                            echo "✅ Validation query successful!<br>";
                            echo "Validation result: " . json_encode($validation_data) . "<br>";
                        } else {
                            echo "❌ Validation query returned no rows<br>";
                            
                            // Test each part of the WHERE clause
                            echo "<strong>Testing WHERE clause parts:</strong><br>";
                            
                            // Test 1: Just token
                            $test1 = $conn->prepare("SELECT COUNT(*) as count FROM admin_sessions WHERE session_token = ?");
                            $test1->bind_param("s", $token);
                            $test1->execute();
                            $count1 = $test1->get_result()->fetch_assoc()['count'];
                            echo "1. Token match: " . $count1 . " rows<br>";
                            
                            // Test 2: Token + expiry
                            $test2 = $conn->prepare("SELECT COUNT(*) as count FROM admin_sessions WHERE session_token = ? AND expires_at > NOW()");
                            $test2->bind_param("s", $token);
                            $test2->execute();
                            $count2 = $test2->get_result()->fetch_assoc()['count'];
                            echo "2. Token + not expired: " . $count2 . " rows<br>";
                            
                            // Test 3: Full query
                            $test3 = $conn->prepare("SELECT COUNT(*) as count FROM admin_sessions s JOIN admin_users a ON s.admin_id = a.id WHERE s.session_token = ? AND s.expires_at > NOW() AND a.is_active = TRUE");
                            $test3->bind_param("s", $token);
                            $test3->execute();
                            $count3 = $test3->get_result()->fetch_assoc()['count'];
                            echo "3. Full JOIN query: " . $count3 . " rows<br>";
                            
                            // Check the admin_users.is_active column type
                            $column_check = $conn->query("DESCRIBE admin_users");
                            echo "<strong>admin_users table structure:</strong><br>";
                            while ($col = $column_check->fetch_assoc()) {
                                if ($col['Field'] == 'is_active') {
                                    echo "is_active column type: " . $col['Type'] . ", Default: " . $col['Default'] . "<br>";
                                }
                            }
                        }
                        if (!$admin_data['is_active']) {
                            echo "❌ PROBLEM: Admin user is INACTIVE!<br>";
                        }
                    } else {
                        echo "❌ PROBLEM: Admin user NOT FOUND in admin_users table!<br>";
                        
                        // Show what admin users exist
                        $all_admins = $conn->query("SELECT id, username, is_active FROM admin_users");
                        echo "<strong>Admin users in database:</strong><br>";
                        while ($row = $all_admins->fetch_assoc()) {
                            echo "- ID: " . $row['id'] . ", Username: " . $row['username'] . ", Active: " . ($row['is_active'] ? 'YES' : 'NO') . "<br>";
                        }
                    }
                } else {
                    echo "❌ Session EXPIRED<br>";
                }
            } else {
                echo "❌ Token NOT found in database<br>";
                
                // Show what tokens exist
                $all_sessions = $conn->query("SELECT session_token, admin_id, expires_at FROM admin_sessions ORDER BY expires_at DESC LIMIT 3");
                echo "<strong>Recent sessions in database:</strong><br>";
                while ($row = $all_sessions->fetch_assoc()) {
                    echo "- " . substr($row['session_token'], 0, 20) . "... (Admin: " . $row['admin_id'] . ", Expires: " . $row['expires_at'] . ")<br>";
                }
            }
            
            $conn->close();
        } catch (Exception $e) {
            echo "❌ Database error: " . $e->getMessage() . "<br>";
        }
        
        $auth = getAdminAuth();
        $admin = $auth->validateSession($_SESSION['admin_session_token']);
        
        if ($admin) {
            echo "✅ Admin validated: " . $admin['full_name'] . "<br>";
            echo "<h2>🎉 SUCCESS! The session works.</h2>";
            echo "<a href='users.php'>Now try users.php</a>";
        } else {
            echo "❌ Session validation failed<br>";
        }
        
    } catch (Exception $e) {
        echo "❌ Error: " . $e->getMessage() . "<br>";
    }
    
} else {
    echo "<h3>❌ No admin session found</h3>";
    echo "<p>Please login first at <a href='login.php'>login.php</a></p>";
}
?>