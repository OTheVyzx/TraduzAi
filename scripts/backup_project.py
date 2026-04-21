import os
import shutil
import datetime
import zipfile
from pathlib import Path

def run_backup():
    root = Path("d:/TraduzAi")
    
    # 1. Update context.md has been done by AI, but we can touch it
    print("Atualizando context.md timestamp...")
    context_file = root / "context.md"
    if context_file.exists():
        content = context_file.read_text(encoding="utf-8")
        # Ensure it's up to date
        context_file.touch()

    # 2. Create new versioned backup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = root / "backups"
    backup_dir.mkdir(exist_ok=True)
    
    new_backup_name = f"traduzai_backup_{timestamp}.zip"
    new_backup_path = backup_dir / new_backup_name
    
    print(f"Criando novo backup: {new_backup_name}...")
    
    # Simple ZIP of src, src-tauri, pipeline, scripts
    folders_to_backup = ["src", "src-tauri", "pipeline", "scripts"]
    files_to_backup = ["package.json", "context.md", "README.md", "AGENTS.md"]
    
    with zipfile.ZipFile(new_backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder in folders_to_backup:
            folder_path = root / folder
            if folder_path.exists():
                for file in folder_path.rglob('*'):
                    if "__pycache__" in str(file) or "node_modules" in str(file) or "target" in str(file):
                        continue
                    zipf.write(file, file.relative_to(root))
        
        for file_name in files_to_backup:
            file_path = root / file_name
            if file_path.exists():
                zipf.write(file_path, file_path.relative_to(root))

    # 3. Delete previous backup
    existing_backups = sorted(list(backup_dir.glob("traduzai_backup_*.zip")))
    if len(existing_backups) > 1:
        # The last one is the one we just created
        to_delete = existing_backups[:-1]
        for old_b in to_delete:
            print(f"Removendo backup anterior: {old_b.name}")
            try:
                old_b.unlink()
            except Exception as e:
                print(f"Erro ao deletar {old_b.name}: {e}")

    print("Backup concluido com sucesso!")

if __name__ == "__main__":
    run_backup()
