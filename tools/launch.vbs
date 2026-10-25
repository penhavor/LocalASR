' LocalASR silent launcher (used by the "LocalASR" shortcut).
' Runs RUN.bat hidden when the environment is ready; visible on the first
' run so the installation progress can be seen.
Option Explicit
Dim fso, sh, appDir, bat, ready
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
appDir = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
bat = appDir & "\RUN.bat"
ready = fso.FileExists(appDir & "\runtime\python\.localasr-env.json") Or _
        fso.FileExists(appDir & "\.venv\.localasr-env.json")
sh.CurrentDirectory = appDir
If ready Then
    sh.Run """" & bat & """", 0, False
Else
    sh.Run """" & bat & """", 1, False
End If
