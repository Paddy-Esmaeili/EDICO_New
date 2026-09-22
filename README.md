# Updated Edico

## Download

### How to download

1. Go to the **[Actions](../../actions)** tab at the top of this repo.
2. Click **Build Executables** in the left sidebar.
3. Click the most recent (top) workflow run — make sure it has a green checkmark ✅.
4. Scroll down to the **Artifacts** section at the bottom of the run page.
5. Download the one for your system:
   - **Edico-Windows** → unzip, then run `Edico.exe`
   - **Edico-Mac** → unzip, then open `Edico.app`
     - On first launch, macOS may block the app since it isn't notarized. Right-click the app → **Open** → **Open** again to bypass Gatekeeper.

> **Note:** You'll need to be signed in to GitHub to download Actions artifacts, and artifacts are automatically deleted after 90 days.

## Building from source

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
python EDICO/main.py
```
