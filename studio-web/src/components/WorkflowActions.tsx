import { ButtonRow, GhostButton, PrimaryButton } from "./chrome";

export function WorkflowActions({
  busy,
  validationMessage,
  onValidate,
  onPreview,
  onRun,
}: {
  busy?: boolean;
  validationMessage: string;
  onValidate: () => void;
  onPreview: () => void;
  onRun: () => void;
}): JSX.Element {
  return (
    <div className="workflow-actions">
      <p className="validation-message">{validationMessage}</p>
      <ButtonRow>
        <GhostButton type="button" onClick={onValidate} disabled={busy}>
          Validate
        </GhostButton>
        <GhostButton type="button" onClick={onPreview} disabled={busy}>
          Preview
        </GhostButton>
        <PrimaryButton type="button" onClick={onRun} disabled={busy}>
          Run
        </PrimaryButton>
      </ButtonRow>
    </div>
  );
}
