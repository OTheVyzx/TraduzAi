export type ToastLevel = "info" | "success" | "error";

export function useToast() {
  return {
    show(message: string, level: ToastLevel = "info") {
      if (level === "error") {
        console.error(message);
        return;
      }
      console.info(message);
    },
  };
}
