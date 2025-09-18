const form = document.getElementById('uploadForm');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const bulletsEl = document.getElementById('bullets');
const docxLink = document.getElementById('docxLink');
const texLink = document.getElementById('texLink');

function addBullet(text){
	const row = document.createElement('div');
	row.className = 'message';
	const avatar = document.createElement('div');
	avatar.className = 'avatar';
	avatar.textContent = '•';
	const bubble = document.createElement('div');
	bubble.className = 'bubble';
	bubble.textContent = text;
	row.appendChild(avatar);
	row.appendChild(bubble);
	bulletsEl.appendChild(row);
}

form.addEventListener('submit', async (e) => {
	e.preventDefault();
	bulletsEl.innerHTML = '';
	resultsEl.classList.add('hidden');
	statusEl.textContent = 'Uploading and processing… this can take a while for long lectures.';

	const file = document.getElementById('video').files[0];
	if (!file) { statusEl.textContent = 'Please choose a file.'; return; }
	const apiKey = document.getElementById('apiKey').value.trim();
	const mode = document.getElementById('mode').value;
	const formData = new FormData();
	formData.append('video', file);

	try {
		const res = await fetch('/api/summarize', {
			method: 'POST',
			body: formData,
			headers: Object.fromEntries(Object.entries({
				'x-gemini-key': apiKey || undefined,
				'x-summarizer-mode': mode,
			}).filter(([_, v]) => v !== undefined)),
		});
		if (!res.ok) throw new Error('Server error');
		const data = await res.json();
		statusEl.textContent = 'Done!';
		resultsEl.classList.remove('hidden');

		bulletsEl.innerHTML = '';
		(data.bullets || []).forEach(addBullet);
		docxLink.href = data.docxUrl;
		docxLink.download = 'summary.docx';
		texLink.href = data.texUrl;
		texLink.download = 'summary.tex';
	} catch (err) {
		console.error(err);
		statusEl.textContent = 'Failed to process the video.';
	}
});
