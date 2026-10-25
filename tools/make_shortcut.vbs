' Creates the "LocalASR" shortcut on the Desktop.
' Target: wscript.exe tools\launch.vbs - starts the app with no console flash.
' Re-running the script simply updates the existing shortcut.
Option Explicit
Dim fso, sh, toolsDir, appDir, ico, lnk
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
toolsDir = fso.GetParentFolderName(WScript.ScriptFullName)
appDir = fso.GetParentFolderName(toolsDir)
ico = appDir & "\resources\app.ico"

Set lnk = sh.CreateShortcut(sh.SpecialFolders("Desktop") & "\LocalASR.lnk")
lnk.TargetPath = sh.ExpandEnvironmentStrings("%SystemRoot%") & "\System32\wscript.exe"
lnk.Arguments = """" & toolsDir & "\launch.vbs"""
lnk.WorkingDirectory = appDir
If fso.FileExists(ico) Then lnk.IconLocation = ico
lnk.Description = "LocalASR - local speech recognition"
lnk.WindowStyle = 1
lnk.Save
WScript.Echo "Shortcut ""LocalASR"" created on the Desktop."
