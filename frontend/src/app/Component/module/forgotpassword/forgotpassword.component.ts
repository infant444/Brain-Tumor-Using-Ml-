import { Component,OnInit} from '@angular/core';
import { UserService } from '../../../Services/user/user.service';
import { tap } from 'rxjs';
import { ToastrService } from 'ngx-toastr';
import { Router } from '@angular/router';

@Component({
  selector: 'app-forgotpassword',
  templateUrl: './forgotpassword.component.html',
  styleUrl: './forgotpassword.component.css'
})
export class ForgotpasswordComponent implements OnInit{
  Email:string="";
  otp:string="";
  sendotp="";
  password:string="";
  confrompassword:string="";
  ispassword=false;
  isotp=false;
  isemail=true;
  validateotp=false;
  validatepassword=false;
  constructor(private userservices:UserService,private toaster:ToastrService,private router:Router){}
  ngOnInit(): void {
  }
  send_otp(){
    console.log(this.Email)
    this.userservices.sendopt(this.Email).subscribe({

        next:(otp)=>{
          console.log(otp);
          this.toaster.success("Successfully email sended");
          this.isemail=false;
          this.isotp=true;
          this.sendotp=otp.otp
        },
        error:(err)=>{
          this.toaster.error(err.error.toString())
        }

  })

  }
  verify(){
    if(this.otp.toString() === this.sendotp.toString()){
      this.isotp=false;
      this.ispassword=true;
      this.validateotp=false;
    }
    else{
      this.validateotp=true;
    }
  }
  changepassword(){
    if((this.password.toString() === this.confrompassword.toString())&&(this.password.toString()!="")&&(this.confrompassword.toString()!="")){
      const x:any={
        password:this.password
      }
      this.userservices.changepassword(this.Email,x).subscribe((user)=>{
        this.toaster.success("Successfully password change");
        this.router.navigateByUrl("/login");
      })
    }else{
      this.validatepassword=true;
    }

  }
}
