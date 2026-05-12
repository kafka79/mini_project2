document.addEventListener('DOMContentLoaded', () => {
    const name1Input = document.getElementById('name1');
    const name2Input = document.getElementById('name2');
    const compareBtn = document.getElementById('compare-btn');
    const resultSection = document.getElementById('result-section');
    const scoreValue = document.getElementById('score-value');
    const scorePath = document.getElementById('score-path');
    const code1 = document.getElementById('code1');
    const code2 = document.getElementById('code2');
    const similarityStatus = document.getElementById('similarity-status');
    const chips = document.querySelectorAll('.chip');

    const API_URL = 'http://localhost:8000';

    async function detectSimilarity() {
        const name1 = name1Input.value.trim();
        const name2 = name2Input.value.trim();

        if (!name1 || !name2) {
            alert('Please enter both names');
            return;
        }

        compareBtn.disabled = true;
        compareBtn.innerHTML = 'Analyzing...';

        try {
            const response = await fetch(`${API_URL}/compare`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name1, name2 })
            });

            if (!response.ok) throw new Error('API Error');

            const data = await response.json();
            updateUI(data);
        } catch (error) {
            console.error(error);
            alert('Could not connect to the backend. Make sure the FastAPI server is running on port 8000.');
        } finally {
            compareBtn.disabled = false;
            compareBtn.innerHTML = 'Detect Similarity <span class="btn-icon">→</span>';
        }
    }

    function updateUI(data) {
        resultSection.classList.remove('hidden');
        
        // Animate score
        let currentScore = 0;
        const targetScore = data.score;
        const duration = 1000;
        const startTime = performance.now();

        function animate(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            currentScore = Math.floor(progress * targetScore);
            scoreValue.textContent = currentScore;
            
            // Update circular progress
            const dashArray = `${currentScore}, 100`;
            scorePath.setAttribute('stroke-dasharray', dashArray);

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        }
        requestAnimationFrame(animate);

        // Update codes
        code1.textContent = data.code1;
        code2.textContent = data.code2;

        // Update status badge
        if (data.score >= 90) {
            similarityStatus.textContent = 'Highly Similar';
            similarityStatus.style.background = 'rgba(16, 185, 129, 0.2)';
            similarityStatus.style.color = '#10b981';
        } else if (data.score >= 70) {
            similarityStatus.textContent = 'Likely Match';
            similarityStatus.style.background = 'rgba(245, 158, 11, 0.2)';
            similarityStatus.style.color = '#f59e0b';
        } else {
            similarityStatus.textContent = 'Distinct Entities';
            similarityStatus.style.background = 'rgba(239, 68, 68, 0.2)';
            similarityStatus.style.color = '#ef4444';
        }
    }

    compareBtn.addEventListener('click', detectSimilarity);

    // Enter key support
    [name1Input, name2Input].forEach(input => {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') detectSimilarity();
        });
    });

    // Sample chips
    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            name1Input.value = chip.dataset.n1;
            name2Input.value = chip.dataset.n2;
            detectSimilarity();
        });
    });
});
