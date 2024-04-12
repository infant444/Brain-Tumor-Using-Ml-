import { Component ,OnInit} from '@angular/core';
import { UserService } from '../../../Services/user/user.service';
import { User } from '../../../model/user';
import { ActivatedRoute } from '@angular/router';
import { BraintumorService } from '../../../Services/brain/braintumor.service';
import { Patient } from '../../../model/patient';
import { AdminService } from '../../../Services/admin/admin.service';

@Component({
  selector: 'app-profile',
  templateUrl: "./profile.component.html",
  styleUrl: './profile.component.css'
})
export class ProfileComponent implements OnInit{
  constructor(private userservice:UserService,private activaterouter:ActivatedRoute,private braintumor:BraintumorService,private adminservices:AdminService){}
  user!:User;
  profile="../../../../assets/images/profile.png";
  report?:Patient[];
  show=true;
ngOnInit(): void {

  this.activaterouter.params.subscribe((params)=>{
    if(params.id){
      this.adminservices.getuser(params.id).subscribe((user)=>{
        this.user=user
        this.show=false
        if(this.user.profile){
          this.profile=this.user.profile;
        }else{
        this.profile="../../../../assets/images/profile.png";

        }
        this.braintumor.getreports(this.user.id).subscribe((report)=>{
          this.report=report;
          console.log(report)
        })
      })
    }
    else{
      this.user=this.userservice.currentUser;
    }
  })


  this.braintumor.getreports(this.user.id).subscribe((report)=>{
    this.report=report;
    console.log(report)
  })

  if(this.user.profile){
    this.profile=this.user.profile;
  }else{
  this.profile="../../../../assets/images/profile.png";

  }

}
logout(){
  this.userservice.logout();
}
changepassword(){

}
}
