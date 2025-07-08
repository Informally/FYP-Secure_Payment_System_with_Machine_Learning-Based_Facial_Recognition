<?php
// admin/classes/AdminAuth.php - Corrected for admin_users table
class AdminAuth {
    private $conn;
    
    public function __construct($db_connection) {
        $this->conn = $db_connection;
    }
    
    /**
     * Authenticate admin user
     */
    public function login($username, $password, $ip_address = null, $user_agent = null) {
        try {
            // CORRECTED: Use admin_users table (your actual table name)
            $stmt = $this->conn->prepare("
                SELECT id, username, email, password_hash, role, full_name, is_active 
                FROM admin_users 
                WHERE (username = ? OR email = ?) AND is_active = 1
            ");
            $stmt->bind_param("ss", $username, $username);
            $stmt->execute();
            $result = $stmt->get_result();
            
            if ($result->num_rows === 0) {
                // Log failed login attempt
                if (function_exists('logAdminSecurityEvent')) {
                    logAdminSecurityEvent($username, 'login_failed', 'failure', [
                        'reason' => 'User not found',
                        'ip' => $ip_address
                    ]);
                }
                return ['success' => false, 'message' => 'Invalid credentials'];
            }
            
            $admin = $result->fetch_assoc();
            
            // Verify password using password_hash field
            if (!password_verify($password, $admin['password_hash'])) {
                if (function_exists('logAdminSecurityEvent')) {
                    logAdminSecurityEvent($admin['username'], 'login_failed', 'failure', [
                        'reason' => 'Invalid password',
                        'ip' => $ip_address,
                        'admin_id' => $admin['id']
                    ]);
                }
                return ['success' => false, 'message' => 'Invalid credentials'];
            }
            
            // Check if admin is active
            if (!$admin['is_active']) {
                return ['success' => false, 'message' => 'Account is not active'];
            }
            
            // Create session - ensure session table exists
            $session_token = $this->generateSessionToken();
            $expires_at = date('Y-m-d H:i:s', strtotime('+8 hours'));
            
            $this->ensureSessionTableExists();
            
            $stmt = $this->conn->prepare("
                INSERT INTO admin_sessions (admin_id, session_token, ip_address, user_agent, expires_at, created_at)
                VALUES (?, ?, ?, ?, ?, NOW())
            ");
            $stmt->bind_param("issss", $admin['id'], $session_token, $ip_address, $user_agent, $expires_at);
            $stmt->execute();
            
            // Update last login in admin_users table
            $stmt = $this->conn->prepare("UPDATE admin_users SET last_login = NOW() WHERE id = ?");
            $stmt->bind_param("i", $admin['id']);
            $stmt->execute();
            
            // Log successful login
            if (function_exists('logAdminSecurityEvent')) {
                logAdminSecurityEvent($admin['username'], 'login_success', 'success', [
                    'admin_id' => $admin['id'],
                    'role' => $admin['role'],
                    'ip' => $ip_address
                ]);
            }
            
            // Clean up old sessions
            $this->cleanupExpiredSessions();
            
            return [
                'success' => true,
                'session_token' => $session_token,
                'admin' => [
                    'id' => $admin['id'],
                    'username' => $admin['username'],
                    'email' => $admin['email'],
                    'role' => $admin['role'],
                    'full_name' => $admin['full_name']
                ]
            ];
            
        } catch (Exception $e) {
            error_log("Admin login error: " . $e->getMessage());
            return ['success' => false, 'message' => 'Login system error'];
        }
    }
    
    /**
     * Validate admin session
     */
    public function validateSession($session_token) {
    try {
        error_log("🔍 validateSession called with token: " . substr($session_token, 0, 20));
        
        // Use admin_users table (your actual table name)
        $stmt = $this->conn->prepare("
            SELECT s.admin_id, s.expires_at, a.username, a.email, a.role, a.full_name, a.is_active
            FROM admin_sessions s
            JOIN admin_users a ON s.admin_id = a.id
            WHERE s.session_token = ? AND s.expires_at > NOW() AND a.is_active = 1
        ");
        $stmt->bind_param("s", $session_token);
        $stmt->execute();
        $result = $stmt->get_result();
        
        error_log("🔍 Query executed, rows found: " . $result->num_rows);
        
        if ($result->num_rows === 0) {
            error_log("❌ No session found or session invalid");
            $stmt->close();
            return false;
        }
        
        $session = $result->fetch_assoc();
        error_log("✅ Session data retrieved: " . json_encode($session));
        
        // Extend session by 2 hours on activity - CHECK IF updated_at COLUMN EXISTS
        $check_updated_at = $this->conn->query("SHOW COLUMNS FROM admin_sessions LIKE 'updated_at'");
        
        if ($check_updated_at->num_rows > 0) {
            // Table has updated_at column
            $new_expires = date('Y-m-d H:i:s', strtotime('+2 hours'));
            $update_stmt = $this->conn->prepare("UPDATE admin_sessions SET expires_at = ?, updated_at = NOW() WHERE session_token = ?");
            $update_stmt->bind_param("ss", $new_expires, $session_token);
        } else {
            // Table doesn't have updated_at column
            $new_expires = date('Y-m-d H:i:s', strtotime('+2 hours'));
            $update_stmt = $this->conn->prepare("UPDATE admin_sessions SET expires_at = ? WHERE session_token = ?");
            $update_stmt->bind_param("ss", $new_expires, $session_token);
        }
        
        $update_result = $update_stmt->execute();
        error_log("🔍 Session update result: " . ($update_result ? 'SUCCESS' : 'FAILED'));
        
        $stmt->close();
        $update_stmt->close();
        
        $return_data = [
            'id' => $session['admin_id'],
            'username' => $session['username'],
            'email' => $session['email'],
            'role' => $session['role'],
            'full_name' => $session['full_name']
        ];
        
        error_log("✅ Returning session data: " . json_encode($return_data));
        return $return_data;
        
    } catch (Exception $e) {
        error_log("💥 validateSession EXCEPTION: " . $e->getMessage());
        error_log("Exception trace: " . $e->getTraceAsString());
        return false;
    }
}
    
    /**
     * Logout admin user
     */
    public function logout($session_token, $admin_id = null) {
        try {
            // Get admin info for logging
            if (!$admin_id) {
                $stmt = $this->conn->prepare("
                    SELECT s.admin_id, a.username 
                    FROM admin_sessions s 
                    JOIN admin_users a ON s.admin_id = a.id 
                    WHERE s.session_token = ?
                ");
                $stmt->bind_param("s", $session_token);
                $stmt->execute();
                $result = $stmt->get_result();
                if ($result->num_rows > 0) {
                    $data = $result->fetch_assoc();
                    $admin_id = $data['admin_id'];
                    $username = $data['username'];
                }
            }
            
            // Delete session
            $stmt = $this->conn->prepare("DELETE FROM admin_sessions WHERE session_token = ?");
            $stmt->bind_param("s", $session_token);
            $stmt->execute();
            
            // Log logout
            if (isset($username) && function_exists('logAdminSecurityEvent')) {
                logAdminSecurityEvent($username, 'logout', 'success', [
                    'admin_id' => $admin_id
                ]);
            }
            
            return true;
            
        } catch (Exception $e) {
            error_log("Logout error: " . $e->getMessage());
            return false;
        }
    }
    
    /**
     * Check if admin has required role
     */
    public function hasRole($admin_role, $required_roles) {
        if (!is_array($required_roles)) {
            $required_roles = [$required_roles];
        }
        
        // Super admin has access to everything
        if ($admin_role === 'super_admin') {
            return true;
        }
        
        return in_array($admin_role, $required_roles);
    }
    
    /**
     * Generate secure session token
     */
    private function generateSessionToken() {
        return bin2hex(random_bytes(32)) . '_' . time();
    }
    
    /**
     * Clean up expired sessions
     */
    private function cleanupExpiredSessions() {
        try {
            $stmt = $this->conn->prepare("DELETE FROM admin_sessions WHERE expires_at < NOW()");
            $stmt->execute();
        } catch (Exception $e) {
            error_log("Session cleanup error: " . $e->getMessage());
        }
    }
    
    /**
     * Ensure admin_sessions table exists
     */
    private function ensureSessionTableExists() {
        try {
            $check_table = $this->conn->query("SHOW TABLES LIKE 'admin_sessions'");
            if ($check_table->num_rows == 0) {
                $create_table = "
                    CREATE TABLE admin_sessions (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        admin_id INT NOT NULL,
                        session_token VARCHAR(100) UNIQUE NOT NULL,
                        ip_address VARCHAR(45),
                        user_agent TEXT,
                        expires_at DATETIME NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_admin_id (admin_id),
                        INDEX idx_session_token (session_token),
                        INDEX idx_expires_at (expires_at),
                        FOREIGN KEY (admin_id) REFERENCES admin_users(id) ON DELETE CASCADE
                    )
                ";
                $this->conn->query($create_table);
                error_log("Created admin_sessions table");
            }
        } catch (Exception $e) {
            error_log("Error creating admin_sessions table: " . $e->getMessage());
        }
    }
    
    /**
     * Create new admin user (super admin only)
     */
    public function createAdminUser($username, $email, $password, $role, $full_name, $created_by_admin_id) {
        try {
            // Check if username/email already exists in admin_users
            $stmt = $this->conn->prepare("SELECT id FROM admin_users WHERE username = ? OR email = ?");
            $stmt->bind_param("ss", $username, $email);
            $stmt->execute();
            if ($stmt->get_result()->num_rows > 0) {
                return ['success' => false, 'message' => 'Username or email already exists'];
            }
            
            // Hash password
            $password_hash = password_hash($password, PASSWORD_DEFAULT);
            
            // Insert into admin_users table
            $stmt = $this->conn->prepare("
                INSERT INTO admin_users (username, email, password_hash, role, full_name, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, 1, NOW())
            ");
            $stmt->bind_param("sssss", $username, $email, $password_hash, $role, $full_name);
            $stmt->execute();
            
            $new_admin_id = $this->conn->insert_id;
            
            // Log admin creation
            if (function_exists('logAdminSecurityEvent')) {
                logAdminSecurityEvent('admin_' . $created_by_admin_id, 'create_admin', 'success', [
                    'new_admin_id' => $new_admin_id,
                    'username' => $username,
                    'role' => $role
                ]);
            }
            
            return ['success' => true, 'admin_id' => $new_admin_id];
            
        } catch (Exception $e) {
            error_log("Create admin error: " . $e->getMessage());
            return ['success' => false, 'message' => 'Failed to create admin user'];
        }
    }
}
?>