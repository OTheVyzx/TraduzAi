use serde::Serialize;
use std::cmp::Ordering;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ImportedPage {
    pub number: u32,
    pub relative_path: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy)]
struct ImportLimits {
    max_pages: usize,
    max_file_bytes: u64,
    max_total_bytes: u64,
    max_archive_entries: usize,
}

impl Default for ImportLimits {
    fn default() -> Self {
        Self {
            max_pages: 2_000,
            max_file_bytes: 100 * 1024 * 1024,
            max_total_bytes: 2 * 1024 * 1024 * 1024,
            max_archive_entries: 10_000,
        }
    }
}

fn prepare_manual_chapter_from_paths(
    source_path: &Path,
    project_json_path: &Path,
    limits: ImportLimits,
) -> Result<Vec<ImportedPage>, String> {
    let source_metadata = fs::symlink_metadata(source_path)
        .map_err(|error| format!("Fonte de imagens indisponível: {error}"))?;
    if source_metadata.file_type().is_symlink() {
        return Err("A fonte não pode ser um link simbólico.".to_string());
    }

    let project_dir = project_json_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or_else(|| "Escolha um caminho completo para project.json.".to_string())?;
    fs::create_dir_all(project_dir)
        .map_err(|error| format!("Não foi possível criar a pasta do capítulo: {error}"))?;

    let destination = project_dir.join("original");
    if fs::symlink_metadata(&destination).is_ok() {
        return Err(format!(
            "A pasta de destino já existe e não será sobrescrita: {}",
            destination.display()
        ));
    }

    let staged_root = project_dir.join(format!(".traduzai-import-{}", Uuid::new_v4()));
    fs::create_dir(&staged_root)
        .map_err(|error| format!("Não foi possível criar a área temporária: {error}"))?;
    let mut staging = StagingGuard::new(staged_root);
    let staged_original = staging.path().join("original");
    fs::create_dir(&staged_original)
        .map_err(|error| format!("Não foi possível preparar a área temporária: {error}"))?;

    let mut files = if source_metadata.is_dir() {
        stage_directory(source_path, &staged_original, limits)?
    } else if source_metadata.is_file() && is_archive(source_path) {
        stage_archive(source_path, &staged_original, limits)?
    } else {
        return Err("Escolha uma pasta, um arquivo ZIP ou um arquivo CBZ.".to_string());
    };

    if files.is_empty() {
        return Err("Nenhuma imagem PNG, JPEG ou WebP foi encontrada.".to_string());
    }

    files.sort_by(|left, right| natural_path_cmp(&left.relative_path, &right.relative_path));
    let mut pages = Vec::with_capacity(files.len());
    for (index, file) in files.iter().enumerate() {
        let reader = image::ImageReader::open(&file.staged_path)
            .map_err(|error| {
                format!(
                    "Não foi possível ler {}: {error}",
                    file.relative_path.display()
                )
            })?
            .with_guessed_format()
            .map_err(|error| {
                format!(
                    "Formato de imagem inválido em {}: {error}",
                    file.relative_path.display()
                )
            })?;
        let (width, height) = reader.into_dimensions().map_err(|error| {
            format!(
                "Imagem inválida em {}: {error}",
                file.relative_path.display()
            )
        })?;
        if width == 0 || height == 0 {
            return Err(format!(
                "Imagem sem dimensões válidas: {}",
                file.relative_path.display()
            ));
        }
        pages.push(ImportedPage {
            number: index as u32 + 1,
            relative_path: format!("original/{}", portable_path(&file.relative_path)),
            width,
            height,
        });
    }

    fs::rename(&staged_original, &destination)
        .map_err(|error| format!("Não foi possível promover a importação validada: {error}"))?;
    staging.commit();
    Ok(pages)
}

#[derive(Debug)]
struct StagedFile {
    relative_path: PathBuf,
    staged_path: PathBuf,
}

struct StagingGuard {
    root: PathBuf,
    committed: bool,
}

impl StagingGuard {
    fn new(root: PathBuf) -> Self {
        Self {
            root,
            committed: false,
        }
    }

    fn path(&self) -> &Path {
        &self.root
    }

    fn commit(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
        self.committed = true;
    }
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}

fn stage_directory(
    source_root: &Path,
    staged_original: &Path,
    limits: ImportLimits,
) -> Result<Vec<StagedFile>, String> {
    let source_root = source_root
        .canonicalize()
        .map_err(|error| format!("Não foi possível validar a pasta de origem: {error}"))?;
    let mut candidates = Vec::new();
    collect_directory_images(&source_root, &source_root, &mut candidates)?;
    validate_candidate_sizes(
        candidates.iter().map(|(_, size)| *size),
        candidates.len(),
        limits,
    )?;

    let mut staged = Vec::with_capacity(candidates.len());
    for (relative_path, _) in candidates {
        let source = source_root.join(&relative_path);
        let destination = safe_destination(staged_original, &relative_path)?;
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("Não foi possível preparar uma subpasta: {error}"))?;
        }
        let mut input = File::open(&source)
            .map_err(|error| format!("Não foi possível abrir {}: {error}", source.display()))?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&destination)
            .map_err(|error| format!("Arquivo duplicado ou inválido na importação: {error}"))?;
        let copied = copy_with_limit(&mut input, &mut output, limits.max_file_bytes)?;
        if copied == 0 {
            return Err(format!("Imagem vazia: {}", source.display()));
        }
        output
            .sync_all()
            .map_err(|error| format!("Falha ao sincronizar imagem: {error}"))?;
        staged.push(StagedFile {
            relative_path,
            staged_path: destination,
        });
    }
    Ok(staged)
}

fn collect_directory_images(
    source_root: &Path,
    current: &Path,
    output: &mut Vec<(PathBuf, u64)>,
) -> Result<(), String> {
    for entry in fs::read_dir(current)
        .map_err(|error| format!("Não foi possível listar {}: {error}", current.display()))?
    {
        let entry = entry.map_err(|error| format!("Entrada de pasta inválida: {error}"))?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| format!("Não foi possível validar {}: {error}", path.display()))?;
        if metadata.file_type().is_symlink() {
            return Err(format!("Link simbólico não permitido: {}", path.display()));
        }
        if metadata.is_dir() {
            collect_directory_images(source_root, &path, output)?;
        } else if metadata.is_file() && is_supported_image(&path) {
            let relative = path
                .strip_prefix(source_root)
                .map_err(|_| "Uma imagem escapou da pasta escolhida.".to_string())?
                .to_path_buf();
            validate_relative_path(&relative)?;
            output.push((relative, metadata.len()));
        }
    }
    Ok(())
}

fn stage_archive(
    source_path: &Path,
    staged_original: &Path,
    limits: ImportLimits,
) -> Result<Vec<StagedFile>, String> {
    let file = File::open(source_path)
        .map_err(|error| format!("Não foi possível abrir o arquivo compactado: {error}"))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|error| format!("ZIP/CBZ inválido: {error}"))?;
    if archive.len() > limits.max_archive_entries {
        return Err(format!(
            "O arquivo compactado excede o limite de {} entradas.",
            limits.max_archive_entries
        ));
    }

    let mut staged = Vec::new();
    let mut total_bytes = 0_u64;
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|error| format!("Entrada compactada inválida: {error}"))?;
        let relative_path = secure_archive_path(entry.name())?;
        if entry.is_dir() {
            continue;
        }
        if entry
            .unix_mode()
            .is_some_and(|mode| mode & 0o170000 == 0o120000)
        {
            return Err(format!(
                "Link simbólico não permitido no arquivo: {}",
                entry.name()
            ));
        }
        if !is_supported_image(&relative_path) {
            continue;
        }
        if staged.len() >= limits.max_pages {
            return Err(format!(
                "A importação excede o limite de {} páginas.",
                limits.max_pages
            ));
        }
        if entry.size() > limits.max_file_bytes {
            return Err(format!(
                "Uma imagem excede o limite de {} bytes.",
                limits.max_file_bytes
            ));
        }
        total_bytes = total_bytes
            .checked_add(entry.size())
            .ok_or_else(|| "O tamanho total da importação excedeu o limite.".to_string())?;
        if total_bytes > limits.max_total_bytes {
            return Err(format!(
                "A importação excede o limite total de {} bytes.",
                limits.max_total_bytes
            ));
        }

        let destination = safe_destination(staged_original, &relative_path)?;
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("Não foi possível preparar uma subpasta: {error}"))?;
        }
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&destination)
            .map_err(|error| format!("Caminho duplicado no arquivo compactado: {error}"))?;
        let copied = copy_with_limit(&mut entry, &mut output, limits.max_file_bytes)?;
        if copied != entry.size() {
            return Err(format!(
                "Tamanho inconsistente ao extrair {}.",
                entry.name()
            ));
        }
        output
            .sync_all()
            .map_err(|error| format!("Falha ao sincronizar imagem: {error}"))?;
        staged.push(StagedFile {
            relative_path,
            staged_path: destination,
        });
    }
    Ok(staged)
}

fn validate_candidate_sizes(
    sizes: impl Iterator<Item = u64>,
    count: usize,
    limits: ImportLimits,
) -> Result<(), String> {
    if count > limits.max_pages {
        return Err(format!(
            "A importação excede o limite de {} páginas.",
            limits.max_pages
        ));
    }
    let mut total = 0_u64;
    for size in sizes {
        if size > limits.max_file_bytes {
            return Err(format!(
                "Uma imagem excede o limite de {} bytes.",
                limits.max_file_bytes
            ));
        }
        total = total
            .checked_add(size)
            .ok_or_else(|| "O tamanho total da importação excedeu o limite.".to_string())?;
        if total > limits.max_total_bytes {
            return Err(format!(
                "A importação excede o limite total de {} bytes.",
                limits.max_total_bytes
            ));
        }
    }
    Ok(())
}

fn copy_with_limit(
    input: &mut impl Read,
    output: &mut impl Write,
    max_bytes: u64,
) -> Result<u64, String> {
    let mut limited = input.take(max_bytes.saturating_add(1));
    let copied = std::io::copy(&mut limited, output)
        .map_err(|error| format!("Falha ao copiar imagem: {error}"))?;
    if copied > max_bytes {
        return Err(format!("Uma imagem excede o limite de {max_bytes} bytes."));
    }
    Ok(copied)
}

fn is_archive(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| matches!(extension.to_ascii_lowercase().as_str(), "zip" | "cbz"))
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "webp"
            )
        })
}

fn secure_archive_path(name: &str) -> Result<PathBuf, String> {
    let normalized = name.replace('\\', "/");
    if normalized.starts_with('/')
        || normalized.starts_with("//")
        || normalized
            .split('/')
            .next()
            .is_some_and(|segment| segment.contains(':'))
        || normalized.split('/').any(|segment| segment == "..")
    {
        return Err(format!("Caminho inseguro no arquivo compactado: {name}"));
    }
    let path = PathBuf::from(normalized);
    validate_relative_path(&path)
        .map_err(|_| format!("Caminho inseguro no arquivo compactado: {name}"))?;
    Ok(path)
}

fn validate_relative_path(path: &Path) -> Result<(), String> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err("Caminho relativo inseguro.".to_string());
    }
    Ok(())
}

fn safe_destination(root: &Path, relative_path: &Path) -> Result<PathBuf, String> {
    validate_relative_path(relative_path)?;
    Ok(root.join(relative_path))
}

fn portable_path(path: &Path) -> String {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(value) => Some(value.to_string_lossy()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn natural_path_cmp(left: &Path, right: &Path) -> Ordering {
    natural_str_cmp(&portable_path(left), &portable_path(right))
}

fn natural_str_cmp(left: &str, right: &str) -> Ordering {
    let left = left.to_lowercase();
    let right = right.to_lowercase();
    let mut left_index = 0;
    let mut right_index = 0;
    let left_bytes = left.as_bytes();
    let right_bytes = right.as_bytes();

    while left_index < left_bytes.len() && right_index < right_bytes.len() {
        if left_bytes[left_index].is_ascii_digit() && right_bytes[right_index].is_ascii_digit() {
            let left_end = digit_end(left_bytes, left_index);
            let right_end = digit_end(right_bytes, right_index);
            let left_digits = &left[left_index..left_end];
            let right_digits = &right[right_index..right_end];
            let left_trimmed = left_digits.trim_start_matches('0');
            let right_trimmed = right_digits.trim_start_matches('0');
            let left_significant = if left_trimmed.is_empty() {
                "0"
            } else {
                left_trimmed
            };
            let right_significant = if right_trimmed.is_empty() {
                "0"
            } else {
                right_trimmed
            };
            let number_order = left_significant
                .len()
                .cmp(&right_significant.len())
                .then_with(|| left_significant.cmp(right_significant));
            if number_order != Ordering::Equal {
                return number_order;
            }
            left_index = left_end;
            right_index = right_end;
            continue;
        }

        let order = left_bytes[left_index].cmp(&right_bytes[right_index]);
        if order != Ordering::Equal {
            return order;
        }
        left_index += 1;
        right_index += 1;
    }
    left_bytes.len().cmp(&right_bytes.len())
}

fn digit_end(bytes: &[u8], start: usize) -> usize {
    let mut index = start;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
    }
    index
}

#[tauri::command]
pub(crate) async fn studio_prepare_manual_chapter(
    source_path: String,
    project_json_path: String,
) -> Result<Vec<ImportedPage>, String> {
    let source_path = PathBuf::from(source_path);
    let project_json_path = PathBuf::from(project_json_path);
    tauri::async_runtime::spawn_blocking(move || {
        prepare_manual_chapter_from_paths(&source_path, &project_json_path, ImportLimits::default())
    })
    .await
    .map_err(|error| format!("Falha ao preparar importação: {error}"))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageFormat};
    use std::fs::{self, File};
    use std::io::Write;
    use zip::write::SimpleFileOptions;

    fn write_image(path: &Path, width: u32, height: u32, format: ImageFormat) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        DynamicImage::new_rgba8(width, height)
            .save_with_format(path, format)
            .unwrap();
    }

    fn project_path(root: &Path) -> PathBuf {
        root.join("chapter").join("project.json")
    }

    #[test]
    fn imports_supported_images_in_natural_order_and_ignores_other_files() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        write_image(&source.join("10.png"), 10, 20, ImageFormat::Png);
        write_image(&source.join("2.jpg"), 2, 3, ImageFormat::Jpeg);
        write_image(&source.join("1.webp"), 4, 5, ImageFormat::WebP);
        fs::write(source.join("notes.txt"), "ignorar").unwrap();

        let pages = prepare_manual_chapter_from_paths(
            &source,
            &project_path(temp.path()),
            ImportLimits::default(),
        )
        .unwrap();

        expect_paths(
            &pages,
            &["original/1.webp", "original/2.jpg", "original/10.png"],
        );
        assert_eq!((pages[0].width, pages[0].height), (4, 5));
        assert!(temp.path().join("chapter/original/1.webp").is_file());
        assert!(!temp.path().join("chapter/original/notes.txt").exists());
    }

    #[test]
    fn rejects_parent_and_absolute_archive_paths() {
        for unsafe_name in ["../escape.png", "C:/escape.png", "/escape.png"] {
            let temp = tempfile::tempdir().unwrap();
            let archive_path = temp.path().join("unsafe.cbz");
            let file = File::create(&archive_path).unwrap();
            let mut archive = zip::ZipWriter::new(file);
            archive
                .start_file(unsafe_name, SimpleFileOptions::default())
                .unwrap();
            archive.write_all(b"not-an-image").unwrap();
            archive.finish().unwrap();

            let error = prepare_manual_chapter_from_paths(
                &archive_path,
                &project_path(temp.path()),
                ImportLimits::default(),
            )
            .unwrap_err();
            assert!(error.contains("inseguro"), "erro inesperado: {error}");
        }
    }

    #[test]
    fn imports_cbz_images_and_preserves_safe_subfolders() {
        let temp = tempfile::tempdir().unwrap();
        let image_path = temp.path().join("source.png");
        write_image(&image_path, 7, 9, ImageFormat::Png);
        let image_bytes = fs::read(&image_path).unwrap();
        let archive_path = temp.path().join("chapter.cbz");
        let file = File::create(&archive_path).unwrap();
        let mut archive = zip::ZipWriter::new(file);
        archive
            .start_file("pages/10.png", SimpleFileOptions::default())
            .unwrap();
        archive.write_all(&image_bytes).unwrap();
        archive
            .start_file("pages/2.png", SimpleFileOptions::default())
            .unwrap();
        archive.write_all(&image_bytes).unwrap();
        archive
            .start_file("metadata.txt", SimpleFileOptions::default())
            .unwrap();
        archive.write_all(b"ignorar").unwrap();
        archive.finish().unwrap();

        let pages = prepare_manual_chapter_from_paths(
            &archive_path,
            &project_path(temp.path()),
            ImportLimits::default(),
        )
        .unwrap();

        expect_paths(&pages, &["original/pages/2.png", "original/pages/10.png"]);
        assert_eq!((pages[0].width, pages[0].height), (7, 9));
        assert!(!temp.path().join("chapter/original/metadata.txt").exists());
    }

    #[test]
    fn enforces_page_file_and_total_limits_and_cleans_staging() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        write_image(&source.join("1.png"), 2, 2, ImageFormat::Png);
        write_image(&source.join("2.png"), 2, 2, ImageFormat::Png);
        let first_size = fs::metadata(source.join("1.png")).unwrap().len();

        let error = prepare_manual_chapter_from_paths(
            &source,
            &project_path(temp.path()),
            ImportLimits {
                max_pages: 1,
                max_file_bytes: first_size,
                max_total_bytes: first_size,
                max_archive_entries: 10,
            },
        )
        .unwrap_err();

        assert!(error.contains("limite"));
        assert!(!temp.path().join("chapter/original").exists());
        let chapter_parent = temp.path().join("chapter");
        if chapter_parent.exists() {
            assert!(fs::read_dir(chapter_parent).unwrap().all(|entry| !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".traduzai-import-")));
        }
    }

    #[test]
    fn removes_staged_files_when_an_image_is_invalid() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        write_image(&source.join("1.png"), 2, 2, ImageFormat::Png);
        fs::write(source.join("2.png"), b"not-a-png").unwrap();

        let error = prepare_manual_chapter_from_paths(
            &source,
            &project_path(temp.path()),
            ImportLimits::default(),
        )
        .unwrap_err();

        assert!(error.contains("Imagem inválida") || error.contains("Formato de imagem inválido"));
        let chapter_parent = temp.path().join("chapter");
        assert!(!chapter_parent.join("original").exists());
        assert!(fs::read_dir(chapter_parent).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".traduzai-import-")));
    }

    #[test]
    fn enforces_individual_total_and_archive_entry_limits() {
        let file_limit_temp = tempfile::tempdir().unwrap();
        let file_source = file_limit_temp.path().join("source");
        write_image(&file_source.join("1.png"), 3, 3, ImageFormat::Png);
        let image_size = fs::metadata(file_source.join("1.png")).unwrap().len();
        let file_error = prepare_manual_chapter_from_paths(
            &file_source,
            &project_path(file_limit_temp.path()),
            ImportLimits {
                max_pages: 10,
                max_file_bytes: image_size - 1,
                max_total_bytes: image_size * 10,
                max_archive_entries: 10,
            },
        )
        .unwrap_err();
        assert!(file_error.contains("Uma imagem excede o limite"));

        let total_limit_temp = tempfile::tempdir().unwrap();
        let total_source = total_limit_temp.path().join("source");
        write_image(&total_source.join("1.png"), 3, 3, ImageFormat::Png);
        write_image(&total_source.join("2.png"), 3, 3, ImageFormat::Png);
        let size_one = fs::metadata(total_source.join("1.png")).unwrap().len();
        let size_two = fs::metadata(total_source.join("2.png")).unwrap().len();
        let total_error = prepare_manual_chapter_from_paths(
            &total_source,
            &project_path(total_limit_temp.path()),
            ImportLimits {
                max_pages: 10,
                max_file_bytes: size_one.max(size_two),
                max_total_bytes: size_one + size_two - 1,
                max_archive_entries: 10,
            },
        )
        .unwrap_err();
        assert!(total_error.contains("limite total"));

        let entry_limit_temp = tempfile::tempdir().unwrap();
        let archive_path = entry_limit_temp.path().join("many.zip");
        let file = File::create(&archive_path).unwrap();
        let mut archive = zip::ZipWriter::new(file);
        archive
            .start_file("one.txt", SimpleFileOptions::default())
            .unwrap();
        archive.write_all(b"1").unwrap();
        archive
            .start_file("two.txt", SimpleFileOptions::default())
            .unwrap();
        archive.write_all(b"2").unwrap();
        archive.finish().unwrap();
        let entry_error = prepare_manual_chapter_from_paths(
            &archive_path,
            &project_path(entry_limit_temp.path()),
            ImportLimits {
                max_pages: 10,
                max_file_bytes: 10,
                max_total_bytes: 10,
                max_archive_entries: 1,
            },
        )
        .unwrap_err();
        assert!(entry_error.contains("limite de 1 entradas"));
    }

    #[test]
    fn rejects_an_existing_original_directory_without_overwriting_it() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        write_image(&source.join("1.png"), 2, 2, ImageFormat::Png);
        let original = temp.path().join("chapter/original");
        fs::create_dir_all(&original).unwrap();
        fs::write(original.join("keep.txt"), "preservar").unwrap();

        let error = prepare_manual_chapter_from_paths(
            &source,
            &project_path(temp.path()),
            ImportLimits::default(),
        )
        .unwrap_err();

        assert!(error.contains("já existe"));
        assert_eq!(
            fs::read_to_string(original.join("keep.txt")).unwrap(),
            "preservar"
        );
    }

    fn expect_paths(pages: &[ImportedPage], expected: &[&str]) {
        assert_eq!(
            pages
                .iter()
                .map(|page| page.relative_path.as_str())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            pages.iter().map(|page| page.number).collect::<Vec<_>>(),
            (1..=pages.len() as u32).collect::<Vec<_>>()
        );
    }
}
