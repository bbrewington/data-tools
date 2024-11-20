Sub RemDuplicate_UserCols()

' 1. Prompt user with Input Message Box --> # columns to work on
' 2. Replace each column ^ with duplicates removed & sorted ascending
'
' **THIS IS DESTRUCTIVE & CAN'T UNDO!  RUN IT ON A COPY OF THE DATA**

Dim i As Integer

MyUserInput = InputBox("Enter # Columns")
CheckGoAhead = InputBox("Are you sure you want to remove duplicates? Cannot undo (1: Yes, 2: No)")

If CheckGoAhead = 1 Then
    For i = 1 To CInt(MyUserInput)
        Application.CutCopyMode = False
        ActiveSheet.Range(Cells(1, i), Cells(65536, i)).RemoveDuplicates Columns:=1, Header:=xlNo
        ActiveSheet.Sort.SortFields.Clear
        ActiveSheet.Sort.SortFields.Add Key:=Range(Cells(1, i), Cells(65536, i)), _
            SortOn:=xlSortOnValues, Order:=xlAscending, DataOption:=xlSortNormal
        With ActiveSheet.Sort
            .SetRange Range(Cells(1, i), Cells(65536, i))
            .Header = xlYes
            .MatchCase = False
            .Orientation = xlTopToBottom
            .SortMethod = xlPinYin
            .Apply
        End With
    Next i

    MsgBox "Remove Duplicates ran on " & MyUserInput & " columns"

Else
    MsgBox "Macro not run"
End If

End Sub
