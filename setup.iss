[Setup]
AppName=Command Runner
AppVersion=1.0
DefaultDirName={autopf}\Command Runner

[Files]

[Run]
Filename: "cmd.exe"; Parameters: "/k {code:CommandToRun}"; StatusMsgCaption: "Running Command"; Description: "Command Runner"; Flags: runminimized

[Code]
function CommandToRun(Param: String): String;
begin
  Result := 'npm install -g npx & npm install --save-dev electron & npm init';
end;