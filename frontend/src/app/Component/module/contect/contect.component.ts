import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { UserService } from '../../../Services/user/user.service';
import { User } from '../../../model/user';
import { ToastrService } from 'ngx-toastr';
import { Clipboard } from '@angular/cdk/clipboard';
import { Contect } from '../../../model/contect';

@Component({
  selector: 'app-contect',
  templateUrl: './contect.component.html',
  styleUrl: './contect.component.css'
})
export class ContectComponent  implements OnInit{

  contect!:FormGroup;
  user!:User;
  name!:string;
  email!:string;
  issubmited:boolean=false;
  supportmail="support@neuroscan.com";
  constructor(private userserviecs:UserService,private formbuilder:FormBuilder,private clipboard:Clipboard,private toastServices:ToastrService){
    this.user=userserviecs.currentUser;
    if(this.user){
      this.name=this.user.name;
      this.email=this.user.email;
    }else{
      this.name='';
      this.email='';
    }
  }

  ngOnInit(): void {
    this.contect=this.formbuilder.group({
      name:[this.name,Validators.required],
      email:[this.email,[Validators.required,Validators.email]],
      message:["",[Validators.required]],
    })
  }
  get contectInfo(){
    return this.contect.controls;
  }
  submit(){
    this.issubmited=true;
    if(this.contect.invalid){
      console.log("x")
      return;
    }
    const value=this.contect.value;
    const contect:Contect={
      id: '',
      name: value.name,
      email: value.email,
      message: value.message,
      date: new Date().toLocaleTimeString('en-US', {
        hour: 'numeric', minute: '2-digit', second: "2-digit", day: "2-digit", month: "2-digit", year: "2-digit", hour12: true
      }).toString(),
      response: ''
    }
    this.userserviecs.sendfeedback(contect).subscribe((x)=>{
      this.toastServices.success("message sended")
    })
  }
  copy(){
    this.clipboard.copy(this.supportmail)
    this.toastServices.success("copyed successfully!..")
  }
}
