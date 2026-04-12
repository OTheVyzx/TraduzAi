interface BootSplashProps {
  progress: number;
  message: string;
}

export function BootSplash({ progress, message }: BootSplashProps) {
  const clamped = Math.max(0.04, Math.min(1, progress));
  const percent = Math.round(clamped * 100);

  return (
    <div className="relative flex h-screen items-center justify-center overflow-hidden bg-[#07070b]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(255,93,128,0.18),_transparent_40%),radial-gradient(circle_at_bottom,_rgba(124,92,255,0.12),_transparent_32%)]" />
      <div className="absolute inset-0 opacity-40 [background-image:linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] [background-size:36px_36px]" />

      <div className="relative w-[min(420px,calc(100vw-40px))] rounded-2xl border border-white/10 bg-black/70 px-8 py-10 shadow-[0_24px_90px_rgba(0,0,0,0.5)] backdrop-blur-xl">
        <div className="mx-auto w-fit rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[10px] uppercase tracking-[0.28em] text-white/45">
          Boot
        </div>

        <h1 className="mt-5 text-center text-4xl font-semibold tracking-tight text-[#ff5d80]">
          TraduzAi
        </h1>
        <p className="mt-3 text-center text-xl text-white/90">{message}</p>

        <div className="mt-8 h-2 overflow-hidden rounded-full bg-white/10">
          <div
            className="h-full rounded-full bg-[linear-gradient(90deg,#ff5d80_0%,#ff7481_35%,#ff8c7b_62%,#d96dff_100%)] shadow-[0_0_20px_rgba(255,93,128,0.35)] transition-all duration-500"
            style={{ width: `${percent}%` }}
          />
        </div>

        <div className="mt-3 flex items-center justify-between text-[11px] text-white/45">
          <span>Inicializando...</span>
          <span>{percent}%</span>
        </div>
      </div>
    </div>
  );
}
