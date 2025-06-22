<?php
// admin/classes/AdminAuth.php
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
            // Get admin user
            $stmt = $this->conn->prepare("
                SELECT id, username, email, password_hash, role, full_name, is_active 
                FROM admin_users 
                WHERE (username = ? OR email = ?) AND is_active = TRUE
            ");
            $stmt->bind_param("ss", $username, $username);
            $stmt->execute();
            $result = $stmt->get_result();
            
            if ($result->num_rows === 0) {
                $this->logAuditAction(null, 'login_failed', 'admin_user', $username, [
                    'reason' => 'User not found',
                    'ip' => $ip_address
                ], $ip_address, $user_agent);
                return ['success' => false, 'message' => 'Invalid credentials'];
            }
            
            $admin = $result->fetch_assoc();
            
            // Verify password
            if (!password_verify($password, $admin['password_hash'])) {
                $this->logAuditAction($admin['id'], 'login_failed', 'admin_user', $admin['id'], [
                    'reason' => 'Invalid password',
                    'ip' => $ip_address
                ], $ip_address, $user_agent);
                return ['success' => false, 'message' => 'Invalid credentials'];
            }
            
            // Create session
            $session_token = $this->generateSessionToken();
            $expires_at = date('Y-m-d H:i:s', strtotime('+8 hours')); // 8-hour session
            
            $stmt = $this->conn->prepare("
                INSERT INTO admin_sessions (admin_id, session_token, ip_address, user_agent, expires_at)
                VALUES (?, ?, ?, ?, ?)
            ");
            $stmt->bind_param("issss", $admin['id'], $session_token, $ip_address, $user_agent, $expires_at);
            $stmt->execute();
            
            // Update last login
            $stmt = $this->conn->prepare("UPDATE admin_users SET last_login = NOW() WHERE id = ?");
            $stmt->bind_param("i", $admin['id']);
            $stmt->execute();
            
            // Log successful login
            $this->logAuditAction($admin['id'], 'login_success', 'admin_user', $admin['id'], [
                'ip' => $ip_address
            ], $ip_address, $user_agent);
            
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
            $stmt = $this->conn->prepare("
                SELECT s.admin_id, s.expires_at, a.username, a.email, a.role, a.full_name, a.is_active
                FROM admin_sessions s
                JOIN admin_users a ON s.admin_id = a.id
                WHERE s.session_token = ? AND s.expires_at > NOW() AND a.is_active = TRUE
            ");
            $stmt->bind_param("s", $session_token);
            $stmt->execute();
            $result = $stmt->get_result();
            
            if ($result->num_rows === 0) {
                return false;
            }
            
            $session = $result->fetch_assoc();
            
            // Extend session by 2 hours on activity
            $new_expires = date('Y-m-d H:i:s', strtotime('+2 hours'));
            $stmt = $this->conn->prepare("UPDATE admin_sessions SET expires_at = ? WHERE session_token = ?");
            $stmt->bind_param("ss", $new_expires, $session_token);
            $stmt->execute();
            
            return [
                'id' => $session['admin_id'],
                'username' => $session['username'],
                'email' => $session['email'],
                'role' => $session['role'],
                'full_name' => $session['full_name']
            ];
            
        } catch (Exception $e) {
            error_log("Session validation error: " . $e->getMessage());
            return false;
        }
    }
    
    /**
     * Logout admin user
     */
    public function logout($session_token, $admin_id = null) {
        try {
            // Get admin_id if not provided
            if (!$admin_id) {
                $stmt = $this->conn->prepare("SELECT admin_id FROM admin_sessions WHERE session_token = ?");
                $stmt->bind_param("s", $session_token);
                $stmt->execute();
                $result = $stmt->get_result();
                if ($result->num_rows > 0) {
                    $admin_id = $result->fetch_assoc()['admin_id'];
                }
            }
            
            // Delete session
            $stmt = $this->conn->prepare("DELETE FROM admin_sessions WHERE session_token = ?");
            $stmt->bind_param("s", $session_token);
            $stmt->execute();
            
            // Log logout
            if ($admin_id) {
                $this->logAuditAction($admin_id, 'logout', 'admin_user', $admin_id, []);
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
     * Log audit action
     */
    public function logAuditAction($admin_id, $action, $target_type, $target_id, $details = [], $ip_address = null, $user_agent = null) {
        try {
            $stmt = $this->conn->prepare("
                INSERT INTO admin_audit_log (admin_id, action, target_type, target_id, details, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ");
            $details_json = json_encode($details);
            $stmt->bind_param("issssss", $admin_id, $action, $target_type, $target_id, $details_json, $ip_address, $user_agent);
            $stmt->execute();
            
        } catch (Exception $e) {
            error_log("Audit log error: " . $e->getMessage());
        }
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
     * Create new admin user (super admin only)
     */
    public function createAdminUser($username, $email, $password, $role, $full_name, $created_by_admin_id) {
        try {
            // Check if username/email already exists
            $stmt = $this->conn->prepare("SELECT id FROM admin_users WHERE username = ? OR email = ?");
            $stmt->bind_param("ss", $username, $email);
            $stmt->execute();
            if ($stmt->get_result()->num_rows > 0) {
                return ['success' => false, 'message' => 'Username or email already exists'];
            }
            
            // Hash password
            $password_hash = password_hash($password, PASSWORD_DEFAULT);
            
            // Insert new admin
            $stmt = $this->conn->prepare("
                INSERT INTO admin_users (username, email, password_hash, role, full_name, created_by)
                VALUES (?, ?, ?, ?, ?, ?)
            ");
            $stmt->bind_param("sssssi", $username, $email, $password_hash, $role, $full_name, $created_by_admin_id);
            $stmt->execute();
            
            $new_admin_id = $this->conn->insert_id;
            
            // Log admin creation
            $this->logAuditAction($created_by_admin_id, 'create_admin', 'admin_user', $new_admin_id, [
                'username' => $username,
                'role' => $role
            ]);
            
            return ['success' => true, 'admin_id' => $new_admin_id];
            
        } catch (Exception $e) {
            error_log("Create admin error: " . $e->getMessage());
            return ['success' => false, 'message' => 'Failed to create admin user'];
        }
    }
}
?>