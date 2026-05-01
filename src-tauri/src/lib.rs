mod commands;
mod export;
pub(crate) mod glossary;
pub(crate) mod internet_context;
pub(crate) mod local_memory;
pub(crate) mod storage;
pub(crate) mod work_context;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            let app_handle = app.handle().clone();
            let storage = crate::storage::service_for_app(&app_handle)?;
            let storage_paths = storage.ensure_base_dirs()?;
            storage.check_writable()?;
            crate::storage::set_configured_paths(storage_paths.clone());

            println!("[TraduzAi] App data: {:?}", storage_paths.root);
            println!("[TraduzAi] Models dir: {:?}", storage_paths.models);

            tauri::async_runtime::spawn(async move {
                if let Err(err) = crate::commands::pipeline::warmup_visual_stack(app_handle).await {
                    eprintln!("[TraduzAi] Warmup de boot falhou: {err}");
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::project::open_file_dialog,
            commands::project::open_source_dialog,
            commands::project::open_multiple_sources_dialog,
            commands::project::open_project_dialog,
            commands::project::save_file_dialog,
            commands::project::export_text_file,
            commands::project::validate_import,
            commands::project::load_project_json,
            commands::project::save_project_json,
            commands::project::load_editor_page,
            commands::project::create_text_layer,
            commands::project::patch_text_layer,
            commands::project::delete_text_layer,
            commands::project::set_layer_visibility,
            commands::project::update_mask_region,
            commands::project::update_brush_region,
            commands::project::export_project,
            commands::project::export_page_psd,
            commands::glossary::load_glossary,
            commands::glossary::save_glossary,
            commands::glossary::upsert_glossary_entry,
            commands::glossary::remove_glossary_entry,
            commands::local_memory::export_local_memory,
            commands::local_memory::import_local_memory,
            commands::local_memory::upsert_memory_work,
            commands::local_memory::record_translation_memory,
            commands::local_memory::record_user_correction,
            commands::local_memory::record_ocr_correction,
            commands::local_memory::suggest_memory_translation,
            commands::pipeline::start_pipeline,
            commands::pipeline::cancel_pipeline,
            commands::pipeline::pause_pipeline,
            commands::pipeline::resume_pipeline,
            commands::pipeline::retypeset_page,
            commands::pipeline::render_preview_page,
            commands::pipeline::reinpaint_page,
            commands::pipeline::process_block,
            commands::pipeline::warmup_visual_stack,
            commands::pipeline::check_gpu,
            commands::pipeline::get_system_profile,
            commands::pipeline::check_models,
            commands::pipeline::download_models,
            commands::pipeline::detect_page,
            commands::pipeline::ocr_page,
            commands::pipeline::translate_page,
            commands::pipeline::search_anilist,
            commands::pipeline::search_work,
            commands::pipeline::enrich_work_context,
            commands::lab::get_lab_state,
            commands::lab::open_lab_window,
            commands::lab::start_lab,
            commands::lab::pause_lab,
            commands::lab::resume_lab,
            commands::lab::stop_lab,
            commands::lab::approve_lab_proposal,
            commands::lab::reject_lab_proposal,
            commands::lab::approve_lab_batch,
            commands::lab::get_lab_reference_preview,
            commands::lab::list_lab_human_feedback,
            commands::lab::pick_lab_source_dir,
            commands::lab::pick_lab_reference_dir,
            commands::lab::pick_lab_source_files,
            commands::lab::pick_lab_reference_files,
            commands::lab::save_lab_human_feedback,
            commands::lab::set_lab_dirs,
            commands::lab::propose_lab_patch,
            commands::lab::export_lab_patch_json,
            commands::lab::apply_lab_patch,
            commands::credits::get_credits,
            commands::settings::save_settings,
            commands::settings::load_settings,
            commands::settings::load_supported_languages,
            commands::settings::check_ollama,
            commands::settings::create_translator_model,
            commands::settings::restart_app,
            commands::storage::get_storage_paths,
            commands::work_context::load_or_create_work_context,
        ])
        .run(tauri::generate_context!())
        .expect("error while running TraduzAi");
}
