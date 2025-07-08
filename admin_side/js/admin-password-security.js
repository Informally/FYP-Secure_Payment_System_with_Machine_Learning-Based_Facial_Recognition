// admin/js/admin-password-security.js - Admin Password Obfuscation
class AdminPasswordSecurity {
    constructor() {
        this.key = "MySecretKey123!@#"; // Same key as customer side
        this.init();
    }

    // XOR obfuscation (same as customer side)
    obfuscate(text) {
        let result = '';
        for (let i = 0; i < text.length; i++) {
            const charCode = text.charCodeAt(i);
            const keyChar = this.key.charCodeAt(i % this.key.length);
            const obfuscated = charCode ^ keyChar;
            result += String.fromCharCode(obfuscated);
        }
        return btoa(result);
    }

    // Make it look encrypted (same as customer side)
    makeItLookEncrypted(obfuscatedData) {
        const randomPrefix = Math.random().toString(36).substring(2, 8);
        const randomSuffix = Math.random().toString(36).substring(2, 6);
        return `ENC_${randomPrefix}_${obfuscatedData}_${randomSuffix}_END`;
    }

    // Honeypot validation
    validateHoneypot() {
        const honeypot = document.querySelector('input[name="website"]');
        return !honeypot || honeypot.value === '';
    }

    // Initialize all password obfuscation on the page
    init() {
        document.addEventListener('DOMContentLoaded', () => {
            this.setupAllForms();
        });
    }

    // Setup obfuscation for all forms on the page
    setupAllForms() {
        const forms = document.querySelectorAll('form');
        forms.forEach(form => this.setupForm(form));
    }

    // Setup individual form
    setupForm(form) {
        form.addEventListener('submit', (e) => {
            // Honeypot check
            if (!this.validateHoneypot()) {
                e.preventDefault();
                alert('Security check failed. Please refresh and try again.');
                return false;
            }

            // Find all password fields and obfuscate them
            this.obfuscatePasswordFields(form);
        });
    }

    // Obfuscate all password fields in a form
    obfuscatePasswordFields(form) {
        // Handle regular password fields
        const passwordFields = form.querySelectorAll('input[type="password"]:not([data-skip-obfuscation])');
        passwordFields.forEach(field => {
            if (field.value) {
                this.obfuscateField(field);
            }
        });
    }

    // Obfuscate a single password field (in-place)
    obfuscateField(field) {
        const originalValue = field.value;
        if (!originalValue) return;

        // Obfuscate the value and replace it directly
        const obfuscated = this.obfuscate(originalValue);
        const encrypted_looking = this.makeItLookEncrypted(obfuscated);
        
        // Simply replace the value - keep the same field name
        field.value = encrypted_looking;
    }

    // Security helpers
    clearSensitiveData() {
        // Clear password fields on page unload
        window.addEventListener('beforeunload', () => {
            document.querySelectorAll('input[type="password"]').forEach(field => {
                field.value = '';
            });
        });
    }

    // Initialize human interaction tracking
    trackHumanInteraction() {
        let humanInteraction = false;
        document.addEventListener('mousedown', () => humanInteraction = true);
        document.addEventListener('keydown', () => humanInteraction = true);

        // Check on form submission
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', (e) => {
                if (!humanInteraction) {
                    e.preventDefault();
                    console.log('🚨 Automated admin submission detected');
                    return false;
                }
            });
        });
    }

    // Admin-specific security logging
    logSecurityEvent(event, details = {}) {
        console.log(`🔐 Admin Security Event: ${event}`, details);
    }
}

// Auto-initialize when script is loaded
const adminPasswordSecurity = new AdminPasswordSecurity();
adminPasswordSecurity.clearSensitiveData();
adminPasswordSecurity.trackHumanInteraction();

// Admin-specific console logging
console.log('🛡️ Admin Password Security initialized');
console.log('🔐 Security features: Password obfuscation, honeypot protection, human interaction tracking');