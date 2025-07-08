// password-security.js - Centralized Password Obfuscation (Clean Version)
class PasswordSecurity {
    constructor() {
        this.key = "MySecretKey123!@#"; // Same key across all pages
        this.init();
    }

    // XOR obfuscation
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

    // Make it look encrypted
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
                alert('Bot detection triggered. Please refresh and try again.');
                return false;
            }

            // Find all password and PIN fields and obfuscate them
            this.obfuscatePasswordFields(form);
        });
    }

    // Obfuscate all password and PIN fields in a form
    obfuscatePasswordFields(form) {
        // Handle regular password fields
        const passwordFields = form.querySelectorAll('input[type="password"]:not([data-skip-obfuscation])');
        passwordFields.forEach(field => {
            if (field.value) {
                this.obfuscateField(field);
            }
        });

        // Handle PIN arrays (like new_pin[])
        const pinArrays = this.groupPinFields(form);
        pinArrays.forEach(pinGroup => {
            this.obfuscatePinArray(pinGroup);
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

    // Group PIN fields by their base name
    groupPinFields(form) {
        const pinFields = form.querySelectorAll('input[name^="new_pin"], input[name^="pin"], input[name="current_pin"]');
        const groups = {};
        
        pinFields.forEach(field => {
            let baseName = field.name;
            
            // Extract base name for arrays like new_pin[0] -> new_pin
            if (baseName.includes('[')) {
                baseName = baseName.substring(0, baseName.indexOf('['));
            }
            
            if (!groups[baseName]) {
                groups[baseName] = [];
            }
            groups[baseName].push(field);
        });
        
        return Object.values(groups);
    }

    // Obfuscate PIN field arrays
    obfuscatePinArray(pinFields) {
        if (pinFields.length === 0) return;
        
        // If it's a single PIN field, just obfuscate it
        if (pinFields.length === 1) {
            this.obfuscateField(pinFields[0]);
            return;
        }
        
        // For multiple PIN fields (like new_pin[0], new_pin[1], etc.)
        const pinValue = pinFields.map(field => field.value).join('');
        if (!pinValue) return;
        
        // Obfuscate each field individually with its digit
        pinFields.forEach(field => {
            if (field.value) {
                const obfuscated = this.obfuscate(field.value);
                const encrypted_looking = this.makeItLookEncrypted(obfuscated);
                field.value = encrypted_looking;
            }
        });
    }

    // Add honeypot to forms that don't have it
    addHoneypot(form) {
        if (!form.querySelector('input[name="website"]')) {
            const honeypot = document.createElement('input');
            honeypot.type = 'text';
            honeypot.name = 'website';
            honeypot.style.display = 'none';
            honeypot.tabIndex = -1;
            honeypot.autocomplete = 'off';
            form.appendChild(honeypot);
        }
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
                    console.log('🚨 Automated submission detected');
                    return false;
                }
            });
        });
    }
}

// Auto-initialize when script is loaded
const passwordSecurity = new PasswordSecurity();
passwordSecurity.clearSensitiveData();
passwordSecurity.trackHumanInteraction();