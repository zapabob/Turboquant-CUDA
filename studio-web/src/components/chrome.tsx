import type { PropsWithChildren, ReactNode } from "react";

export function Panel({
  title,
  subtitle,
  actions,
  children,
}: PropsWithChildren<{
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}>): JSX.Element {
  return (
    <section className="panel">
      <header className="panel-header">
        <div>
          <p className="panel-title">{title}</p>
          {subtitle ? <p className="panel-subtitle">{subtitle}</p> : null}
        </div>
        {actions ? <div className="panel-actions">{actions}</div> : null}
      </header>
      {children}
    </section>
  );
}

export function StatusPill({
  tone,
  children,
}: PropsWithChildren<{ tone: "ok" | "bad" | "warn" | "run" | "idle" }>): JSX.Element {
  return <span className={`pill pill-${tone}`}>{children}</span>;
}

export function Field({
  label,
  helper,
  children,
}: PropsWithChildren<{ label: string; helper?: string }>): JSX.Element {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {helper ? <span className="field-helper">{helper}</span> : null}
      {children}
    </label>
  );
}

export function ButtonRow({ children }: PropsWithChildren): JSX.Element {
  return <div className="button-row">{children}</div>;
}

export function PrimaryButton(
  props: React.ButtonHTMLAttributes<HTMLButtonElement>,
): JSX.Element {
  return <button {...props} className={["primary-button", props.className].filter(Boolean).join(" ")} />;
}

export function GhostButton(
  props: React.ButtonHTMLAttributes<HTMLButtonElement>,
): JSX.Element {
  return <button {...props} className={["ghost-button", props.className].filter(Boolean).join(" ")} />;
}
