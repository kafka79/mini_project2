document.addEventListener('DOMContentLoaded', () => {
    const name1Input = document.getElementById('name1');
    const name2Input = document.getElementById('name2');
    const name1Error = document.getElementById('name1-error');
    const name2Error = document.getElementById('name2-error');
    const compareBtn = document.getElementById('compare-btn');
    const btnText = document.getElementById('btn-text');
    const btnSpinner = document.getElementById('btn-spinner');
    const btnArrow = document.getElementById('btn-arrow');
    const resultSection = document.getElementById('result-section');
    const scoreValue = document.getElementById('score-value');
    const scorePath = document.getElementById('score-path');
    const code1 = document.getElementById('code1');
    const code2 = document.getElementById('code2');
    const similarityStatus = document.getElementById('similarity-status');
    const chips = document.querySelectorAll('.chip');
    const toastContainer = document.getElementById('toast-container');
    
    // Theme toggle & Advanced Options
    const themeToggle = document.getElementById('theme-toggle');
    const thresholdSlider = document.getElementById('threshold-slider');
    const thresholdVal = document.getElementById('threshold-val');
    const aliasesCheckbox = document.getElementById('aliases-checkbox');

    let animationFrameId = null;

    // --- Light/Dark Theme Management ---
    const savedTheme = localStorage.getItem('theme') || 'dark';
    if (savedTheme === 'light') {
        document.body.classList.add('light-theme');
        themeToggle.textContent = '☀️';
    } else {
        themeToggle.textContent = '🌙';
    }

    themeToggle.addEventListener('click', () => {
        const isLight = document.body.classList.toggle('light-theme');
        if (isLight) {
            themeToggle.textContent = '☀️';
            localStorage.setItem('theme', 'light');
        } else {
            themeToggle.textContent = '🌙';
            localStorage.setItem('theme', 'dark');
        }
    });

    // --- Dynamic Slider Text ---
    thresholdSlider.addEventListener('input', (e) => {
        thresholdVal.textContent = `${e.target.value}%`;
    });

    // Remove error highlights on user input
    name1Input.addEventListener('input', () => {
        name1Input.classList.remove('error');
        name1Error.classList.add('hidden');
    });
    name2Input.addEventListener('input', () => {
        name2Input.classList.remove('error');
        name2Error.classList.add('hidden');
    });

    // Toast Notification helper
    function showToast(message, type = 'error') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        const messageSpan = document.createElement('span');
        messageSpan.className = 'toast-message';
        messageSpan.textContent = message;

        const closeBtn = document.createElement('button');
        closeBtn.className = 'toast-close';
        closeBtn.innerHTML = '&times;';

        toast.appendChild(messageSpan);
        toast.appendChild(closeBtn);
        toastContainer.appendChild(toast);

        // Slide/Fade in
        setTimeout(() => toast.classList.add('show'), 10);

        // Auto remove
        const autoClose = setTimeout(() => closeToast(toast), 4000);

        // Manual close
        toast.querySelector('.toast-close').addEventListener('click', () => {
            clearTimeout(autoClose);
            closeToast(toast);
        });
    }

    function closeToast(toast) {
        toast.classList.remove('show');
        toast.addEventListener('transitionend', () => {
            toast.remove();
        });
    }

    async function detectSimilarity() {
        const name1 = name1Input.value.trim();
        const name2 = name2Input.value.trim();
        const customThreshold = parseFloat(thresholdSlider.value);
        const enableAliases = aliasesCheckbox.checked;

        // Reset error styling
        name1Input.classList.remove('error');
        name2Input.classList.remove('error');
        name1Error.classList.add('hidden');
        name2Error.classList.add('hidden');

        let hasError = false;
        if (!name1) {
            name1Input.classList.add('error');
            name1Error.textContent = 'This field is required';
            name1Error.classList.remove('hidden');
            hasError = true;
        }
        if (!name2) {
            name2Input.classList.add('error');
            name2Error.textContent = 'This field is required';
            name2Error.classList.remove('hidden');
            hasError = true;
        }

        if (hasError) {
            return;
        }

        // Toggle Loading States
        setLoading(true);

        try {
            const response = await fetch('/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    name1, 
                    name2, 
                    enable_aliases: enableAliases, 
                    threshold: customThreshold 
                })
            });

            const data = await response.json();

            if (!response.ok) {
                const errMsg = data.detail || 'An error occurred during evaluation.';
                if (data.detail && data.detail.includes('First')) {
                    name1Input.classList.add('error');
                    name1Error.textContent = errMsg;
                    name1Error.classList.remove('hidden');
                } else if (data.detail && data.detail.includes('Second')) {
                    name2Input.classList.add('error');
                    name2Error.textContent = errMsg;
                    name2Error.classList.remove('hidden');
                } else {
                    name1Input.classList.add('error');
                    name2Input.classList.add('error');
                    showToast(errMsg);
                }
                throw new Error(errMsg);
            }

            updateUI(data, customThreshold);
        } catch (error) {
            console.error('API Error:', error);
            showToast(error.message || 'Could not connect to the backend.');
        } finally {
            setLoading(false);
        }
    }

    function setLoading(isLoading) {
        compareBtn.disabled = isLoading;
        name1Input.disabled = isLoading;
        name2Input.disabled = isLoading;

        if (isLoading) {
            btnText.textContent = 'Analyzing...';
            btnSpinner.classList.remove('hidden');
            btnArrow.classList.add('hidden');
        } else {
            btnText.textContent = 'Detect Similarity';
            btnSpinner.classList.add('hidden');
            btnArrow.classList.remove('hidden');
        }
    }

    function updateUI(data, customThreshold) {
        resultSection.classList.remove('hidden');
        
        const targetScore = data.score;
        const duration = 800; // ms
        const startTime = performance.now();

        // 1. Reset semantic match classes on circular path using design system rules
        scorePath.classList.remove('stroke-high', 'stroke-med', 'stroke-low');
        if (targetScore >= customThreshold) {
            if (targetScore >= 90) {
                scorePath.classList.add('stroke-high');
            } else {
                scorePath.classList.add('stroke-med');
            }
        } else {
            scorePath.classList.add('stroke-low');
        }

        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
        }

        function animate(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easeProgress = progress * (2 - progress); // easeOutQuad
            const currentScore = Math.floor(easeProgress * targetScore);
            scoreValue.textContent = currentScore;
            
            const dashArray = `${currentScore}, 100`;
            scorePath.setAttribute('stroke-dasharray', dashArray);

            if (progress < 1) {
                animationFrameId = requestAnimationFrame(animate);
            }
        }
        animationFrameId = requestAnimationFrame(animate);

        code1.textContent = data.code1 || 'N/A';
        code2.textContent = data.code2 || 'N/A';

        // 2. Reset semantic badge styles using CSS Classes
        similarityStatus.classList.remove('status-high', 'status-med', 'status-low');
        if (data.score >= customThreshold) {
            if (data.score >= 90) {
                similarityStatus.textContent = data.match_type === 'alias' ? 'Verified Alias' : 'Highly Similar';
                similarityStatus.classList.add('status-high');
            } else {
                similarityStatus.textContent = 'Likely Match';
                similarityStatus.classList.add('status-med');
            }
        } else {
            similarityStatus.textContent = 'Distinct Entities';
            similarityStatus.classList.add('status-low');
        }
    }

    compareBtn.addEventListener('click', detectSimilarity);

    [name1Input, name2Input].forEach(input => {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !compareBtn.disabled) {
                detectSimilarity();
            }
        });
    });

    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            if (compareBtn.disabled) return;
            name1Input.value = chip.dataset.n1;
            name2Input.value = chip.dataset.n2;
            detectSimilarity();
        });
    });
});
