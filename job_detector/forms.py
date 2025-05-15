from django import forms

class JobPostForm(forms.Form):
    job_text = forms.CharField(widget=forms.Textarea(attrs={'rows': 6}), required=False)
    job_file = forms.FileField(required=False)
    job_url = forms.URLField(required=False)
