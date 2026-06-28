import { SignIn } from "@clerk/nextjs";

export default function SignInPage() {
  return (
    <div className="min-h-screen bg-[#0A0A0F] flex items-center justify-center p-6">
      <SignIn
        routing="path"
        path="/sign-in"
        afterSignInUrl="/dashboard"
        appearance={{
          variables: {
            colorPrimary: "#3B82F6",
            colorBackground: "#111116",
            colorInputBackground: "#18181B",
            colorInputText: "#FFFFFF",
            colorText: "#FFFFFF",
            colorTextSecondary: "#A1A1AA",
            borderRadius: "12px",
          },
          elements: {
            card: "bg-[#111116] border border-white/10 shadow-2xl",
            headerTitle: "text-3xl font-bold text-white",
            headerSubtitle: "text-zinc-400",
            formButtonPrimary:
              "bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500",
            socialButtonsBlockButton:
              "bg-zinc-900 border border-zinc-700 hover:bg-zinc-800",
            formFieldInput:
              "bg-zinc-900 border border-zinc-700 text-white",
            footerActionLink:
              "text-blue-400 hover:text-blue-300",
          },
        }}
      />
    </div>
  );
}