const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;

const uploadsDir = path.join(process.cwd(), 'uploads');
const outputsDir = path.join(process.cwd(), 'outputs');
const publicDir = path.join(process.cwd(), 'public');

for (const dir of [uploadsDir, outputsDir, publicDir]) {
	if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

app.use(express.static(publicDir));
app.use('/outputs', express.static(outputsDir));

const storage = multer.diskStorage({
	destination: (req, file, cb) => cb(null, uploadsDir),
	filename: (req, file, cb) => {
		const ext = path.extname(file.originalname);
		const base = path.basename(file.originalname, ext);
		cb(null, `${base}_${Date.now()}${ext}`);
	},
});
const upload = multer({ storage });

app.post('/api/summarize', upload.single('video'), async (req, res) => {
	try {
		if (!req.file) return res.status(400).json({ error: 'No file uploaded' });

		const python = process.env.PYTHON_BIN || 'python';
		const args = [
			'main.py',
			'--input-dir', 'uploads',
			'--output-dir', 'outputs',
			'--model-size', 'base',
			'--gemini-model', 'gemini-1.5-flash',
			'--web-single',
		];

		const env = { ...process.env };
		if (req.headers['x-gemini-key']) env.GEMINI_API_KEY = req.headers['x-gemini-key'];
		if (req.headers['x-summarizer-mode']) env.SUMMARIZER_MODE = String(req.headers['x-summarizer-mode']).toLowerCase();

		await runPython(python, args, env);

		const jsonPath = path.join(outputsDir, 'summary.json');
		const bullets = fs.existsSync(jsonPath)
			? JSON.parse(fs.readFileSync(jsonPath, 'utf-8')).bullets || []
			: [];

		return res.json({
			bullets,
			docxUrl: '/outputs/summary.docx',
			texUrl: '/outputs/summary.tex',
		});
	} catch (err) {
		console.error(err);
		return res.status(500).json({ error: 'Processing failed', detail: String(err) });
	}
});

function runPython(python, args, env) {
	return new Promise((resolve, reject) => {
		const proc = spawn(python, args, { env, stdio: 'inherit' });
		proc.on('error', reject);
		proc.on('close', (code) => {
			if (code === 0) resolve();
			else reject(new Error(`Python process exited with code ${code}`));
		});
	});
}

app.listen(PORT, () => {
	console.log(`Server listening on http://localhost:${PORT}`);
});