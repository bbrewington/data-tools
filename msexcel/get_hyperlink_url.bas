' Function to extract hyperlink from a cell
' I wrote this for links of format https://domain.here.com/#/endpoint-here
' The code was interpreting the hash "#" sign as a fragment identifier
' More info on URL fragments here: https://www.w3.org/TR/WD-html40-970708/htmlweb.html#h-4.1.1:~:text=%5BRFC1738%5D.-,Fragment%20URLs,-The%20URL%20specification

Function ExtractFullHL(cell As Range) As String
    On Error Resume Next
    Dim fullAddress As String
    If cell.Hyperlinks.Count > 0 Then
        fullAddress = cell.Hyperlinks(1).Address
        If cell.Hyperlinks(1).SubAddress <> "" Then
            fullAddress = fullAddress & "#" & cell.Hyperlinks(1).SubAddress
        End If
    End If
    ExtractFullHL = fullAddress
    On Error GoTo 0
End Function