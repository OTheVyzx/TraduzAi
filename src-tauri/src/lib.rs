mod commands;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            let app_data = std::path::PathBuf::from("D:\\traduzai_data");
            let models_dir = app_data.join("models");
            let projects_dir = app_data.join("projects");

            std::fs::create_dir_all(&models_dir).ok();
            std::fs::create_dir_all(&projects_dir).ok();

            println!("[TraduzAi] App data: {:?}", app_data);
            println!("[TraduzAi] Models dir: {:?}", models_dir);

            let app_handle = app.handle().clone();
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
            commands::pipeline::start_pipeline,
            commands::pipeline::cancel_pipeline,
            commands::pipeline::pause_pipeline,
            commands::pipeline::resume_pipeline,
            commands::pipeline::retypeset_page,
            commands::pipeline::reinpaint_page,
            commands::pipeline::process_block,
            commands::pipeline::warmup_visual_stack,
            commands::pipeline::check_gpu,
            commands::pipeline::get_system_profile,
            commands::pipeline::check_models,
            commands::pipeline::download_models,
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
            commands::lab::pick_lab_source_dir,
            commands::lab::pick_lab_reference_dir,
            commands::lab::pick_lab_source_files,
            commands::lab::pick_lab_reference_files,
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running TraduzAi");
}
